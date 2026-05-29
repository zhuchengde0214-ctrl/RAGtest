"""PlannerAgent：LLM 决策执行计划。

输入：state.user_request（自然语言需求）+ 当前已存在的 state（如已有 parsed_doc 就跳过 parser）
输出：state.plan（list[str]）+ state.plan_reasoning（解释）

可用 agent 清单（与 LangGraphOrchestrator 的 AGENT_REGISTRY 对齐）：
  - parser  : PDF 解析
  - indexer : 分块 + 建索引
  - qa      : 三类 RAG 问答
  - audit   : 10 维度合同审查
  - diff    : 跨合同对比（需要 pdf_path_v2）

PlannerAgent 不会真的执行其它 agent，它只输出 plan，由 Orchestrator / LangGraph
按 plan 调度后续节点。
"""

import json
import logging
import os
import re
from typing import Optional

from llm_client import get_default_model, make_llm_client
from qa_engine import _escape_inner_quotes, _repair_truncated_json

from .base import BaseAgent
from .state import SharedState

logger = logging.getLogger(__name__)


AVAILABLE_AGENTS = {
    "parser": "PDF 解析（Vision OCR + 跨页表格合并）",
    "indexer": "对解析结果做分块并建立 dense+BM25 检索索引",
    "qa": "跑题目固定的三类问答（Q1 简单事实 / Q2 多轮 / Q3 复杂推理）",
    "audit": "做 10 维度合同风险审查（金额、付款、违约、数据安全等）",
    "diff": "对比 v1 与 v2 两份合同的条款差异（需要 pdf_path_v2）",
    "reflection": "审视 audit 输出的风险列表，发现重复/不一致/遗漏并触发重跑（必须紧跟 audit 之后）",
}

PROMPT_TPL = """你是一个工作流规划器。根据用户需求，从可用 agent 中挑出**必要且按顺序**的 agent。

可用 agent：
{agents_desc}

依赖规则（必须遵守）：
- qa / audit / diff 都依赖 indexer
- indexer 依赖 parser
- diff 还要求 user 已经提供了 pdf_path_v2

当前已就绪的产物（已就绪就不要重复再选）：
{state_status}

用户需求：{user_request}

请只输出一个 JSON 对象，不要 ```：
{{
  "plan": ["agent_name_in_order"],
  "reasoning": "为什么这么选（≤80 字）"
}}

要求：
1. plan 至少 1 个 agent
2. plan 必须满足上面的依赖规则
3. 不要把已经就绪的 agent 重复加入 plan"""


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self):
        super().__init__()
        self.client = make_llm_client()
        self.model = get_default_model()

    def _run(self, state: SharedState) -> None:
        # 默认 plan：用户没有 request 时给全套
        if not state.user_request.strip():
            state.plan = self._default_plan(state)
            state.plan_reasoning = "未提供 --request，使用默认全套流水线"
            state.log(self.name, f"使用默认 plan：{state.plan}")
            return

        # 调 LLM 生成 plan
        prompt = PROMPT_TPL.format(
            agents_desc=self._format_agents(),
            state_status=self._format_state(state),
            user_request=state.user_request.strip(),
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
        except Exception as e:
            logger.warning(f"PlannerAgent LLM 调用失败：{e}，回退默认 plan")
            state.plan = self._default_plan(state)
            state.plan_reasoning = f"LLM 调用失败，回退默认：{e}"
            return

        plan, reasoning = self._parse(raw)

        # 校验：依赖规则 + 已就绪 agent 跳过
        plan = self._validate_and_fix(plan, state)

        state.plan = plan
        state.plan_reasoning = reasoning
        state.log(
            self.name,
            f"plan 已生成",
            plan=plan,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _format_agents() -> str:
        return "\n".join(f"  - {k}: {v}" for k, v in AVAILABLE_AGENTS.items())

    @staticmethod
    def _format_state(state: SharedState) -> str:
        from pathlib import Path
        ready = []
        if state.parsed_doc is not None:
            ready.append("parser（已有 parsed_doc）")
        if state.retriever is not None:
            ready.append("indexer（已有 retriever）")
        if state.qa_results:
            ready.append(f"qa（已有 {len(state.qa_results)} 条结果）")
        if state.risks:
            ready.append(f"audit（已有 {len(state.risks)} 条风险）")
        # v2 既可来自 pdf_path_v2（真实 PDF），也可来自合成的 parsed_document_v2.json
        v2_cached = Path(state.output_dir or "outputs", "parsed_document_v2.json").exists()
        has_v2 = bool(state.pdf_path_v2) or v2_cached
        v2_str = f"是（{'PDF' if state.pdf_path_v2 else '已落盘的解析数据'}）" if has_v2 else "否"
        return (
            f"  - 已就绪：{', '.join(ready) if ready else '（无）'}\n"
            f"  - 是否提供了 v2 合同：{v2_str}"
        )

    @staticmethod
    def _parse(raw: str) -> tuple[list[str], str]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        attempts = [text]
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            attempts.append(m.group(0))

        for cand in attempts:
            for variant in (cand, _escape_inner_quotes(cand)):
                try:
                    data = json.loads(variant)
                    plan = data.get("plan", [])
                    reasoning = data.get("reasoning", "")
                    if isinstance(plan, list):
                        return [str(x) for x in plan], str(reasoning)
                except Exception:
                    pass

        # 截断兜底
        repaired = _repair_truncated_json(text)
        if repaired:
            try:
                data = json.loads(_escape_inner_quotes(repaired))
                return data.get("plan", []), data.get("reasoning", "")
            except Exception:
                pass

        logger.warning(f"PlannerAgent 输出无法解析：{raw[:200]}")
        return [], "(LLM 输出无法解析，将回退默认 plan)"

    @staticmethod
    def _default_plan(state: SharedState) -> list[str]:
        plan = []
        if state.parsed_doc is None:
            plan.append("parser")
        if state.retriever is None:
            plan.append("indexer")
        plan.extend(["qa", "audit", "reflection"])
        if state.pdf_path_v2 and "diff" not in plan:
            plan.append("diff")
        return plan

    @staticmethod
    def _validate_and_fix(plan: list[str], state: SharedState) -> list[str]:
        """剔除未知 agent；按依赖关系补齐前置；剔除已就绪的 agent。"""
        valid = [a for a in plan if a in AVAILABLE_AGENTS]
        if not valid:
            return PlannerAgent._default_plan(state)

        # 依赖图（简单显式声明）
        deps = {
            "indexer":     ["parser"],
            "qa":          ["parser", "indexer"],
            "audit":       ["parser", "indexer"],
            "diff":        ["parser", "indexer"],
            "reflection":  ["parser", "indexer", "audit"],
        }

        # 已就绪的 agent
        ready = set()
        if state.parsed_doc is not None:
            ready.add("parser")
        if state.retriever is not None:
            ready.add("indexer")
        # qa / audit 即便已经跑过，也允许再跑（用户可能想刷新结果）

        # 拓扑展开：把每个 agent 的所有未就绪依赖前置补齐
        out: list[str] = []
        seen = set()
        for a in valid:
            for d in deps.get(a, []):
                if d in ready:
                    continue
                if d not in seen:
                    out.append(d)
                    seen.add(d)
            if a not in seen:
                out.append(a)
                seen.add(a)

        # diff 依赖 v2（PDF 或合成的 v2 JSON 都可以）
        from pathlib import Path
        v2_cached = Path(state.output_dir or "outputs", "parsed_document_v2.json").exists()
        has_v2 = bool(state.pdf_path_v2) or v2_cached
        if "diff" in out and not has_v2:
            out = [a for a in out if a != "diff"]

        # 默认补 reflection：只要有 audit 且用户没明确说"不要反思"
        if "audit" in out and "reflection" not in out:
            out.append("reflection")

        return out


def explain_plan(plan: list[str]) -> str:
    """给 trace / log 用的人类可读 plan 描述"""
    return " → ".join(plan)
