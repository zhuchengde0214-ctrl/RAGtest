"""ReflectionAgent：审视 audit 风险列表的质量。

目标：
1. 检查重复风险（不同 risk_id 但描述/证据高度重叠）→ 提示去重
2. 检查严重度异常（同一证据的两条风险一个 high 一个 low）→ 提示统一
3. 检查遗漏（用户 request 提到的关注点是否在 risks 中体现）→ 提示重跑

输出：
- state.reflection_notes（list[str]）：人类可读的审视结论
- state.needs_rerun：None 或 "audit"（暂只支持 audit 重跑）
- state.reflection_iters += 1

ReAct 循环由 LangGraph 条件边实现（阶段 5 改造 graph）：
   audit → reflection → [needs_rerun? → audit | END]

重跑次数硬上限 = 1（避免无限循环）；reflection 把 notes 写进 state，
audit 重跑时看到 state.reflection_notes 会调整 prompt（这次先简单：把 notes
作为 system 级提示拼到 prompt 头）。
"""

import json
import logging
import re
from typing import Optional

from llm_client import get_default_model, make_llm_client
from qa_engine import _escape_inner_quotes, _repair_truncated_json

from .base import BaseAgent
from .state import SharedState

logger = logging.getLogger(__name__)


MAX_REFLECTION_ITERS = 1


REFLECTION_PROMPT = """你是合同审查的 quality reviewer。我们刚跑完一次 10 维度合同风险审查，下面是输出的风险清单（精简版）。

## 风险清单（共 {n_risks} 条）
{risks_text}

## 用户原始需求（如有）
{user_request}

请审视这份风险清单，找出三类问题：
1. **重复**：不同 risk_id 但描述或证据高度重叠 → 标记为 duplicate
2. **严重度不一致**：相似的风险被标了不同的 severity → 标记为 severity_inconsistent
3. **可能遗漏**：用户需求/常识告诉我们应该有但清单里没有的风险点 → 标记为 missing

输出 JSON 对象（不要 ```）：
{{
  "duplicates": [["R001", "R005"], ...],   // 每个内部 list 是同一组重复
  "severity_inconsistent": [
    {{"risk_ids": ["R002","R007"], "note": "为什么不一致"}}
  ],
  "missing": [
    {{"topic": "应该但没有的风险点", "reason": "依据"}}
  ],
  "needs_rerun": true | false,             // 如果 missing 非空且不可忽略 → true
  "rerun_focus": "若 needs_rerun=true，描述重跑 audit 时应额外关注什么（≤120 字）"
}}

判断准则：
- duplicates / severity_inconsistent 仅作记录，不触发重跑
- 仅当 missing 至少 1 条且评估为重要时，needs_rerun=true
- 已经做过一次反思（reflection_iters≥{max_iters}）时，needs_rerun 务必为 false（即便仍有 missing）"""


class ReflectionAgent(BaseAgent):
    name = "reflection"

    def __init__(self):
        super().__init__()
        self.client = make_llm_client()
        self.model = get_default_model()

    def check_preconditions(self, state):
        if not state.risks:
            return "无风险列表可审视（先跑 audit）"
        return None

    def _run(self, state: SharedState) -> None:
        risks_text = self._format_risks(state.risks)
        prompt = REFLECTION_PROMPT.format(
            n_risks=len(state.risks),
            risks_text=risks_text,
            user_request=state.user_request or "(未提供)",
            max_iters=MAX_REFLECTION_ITERS,
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
        except Exception as e:
            logger.warning(f"ReflectionAgent LLM 调用失败：{e}")
            state.log(self.name, f"调用失败，跳过反思：{e}", level="warning")
            state.needs_rerun = None
            return

        result = self._parse(raw)

        notes = self._format_notes(result)
        state.reflection_notes.extend(notes)

        # 决定是否重跑
        if state.reflection_iters >= MAX_REFLECTION_ITERS:
            # 已达上限，强制不再重跑
            state.needs_rerun = None
            state.log(
                self.name,
                f"已达反思次数上限 {MAX_REFLECTION_ITERS}，不再重跑",
                duplicates=len(result.get("duplicates", [])),
                missing=len(result.get("missing", [])),
            )
            return

        if result.get("needs_rerun"):
            state.needs_rerun = "audit"
            state.reflection_iters += 1
            state.log(
                self.name,
                "建议重跑 audit",
                rerun_focus=result.get("rerun_focus", ""),
                missing=len(result.get("missing", [])),
            )
        else:
            state.needs_rerun = None
            state.log(
                self.name,
                "审视通过，无需重跑",
                duplicates=len(result.get("duplicates", [])),
                severity_inconsistent=len(result.get("severity_inconsistent", [])),
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _format_risks(risks: list[dict]) -> str:
        lines = []
        for r in risks:
            ev_first = r.get("evidence", [{}])[0] if r.get("evidence") else {}
            lines.append(
                f"- [{r.get('risk_id','?')}] sev={r.get('severity','?')} "
                f"type={r.get('risk_type','?')} | {r.get('title','')} "
                f"| evidence: {ev_first.get('source_id','?')}@{ev_first.get('section','')[:30]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        for cand in (text, _escape_inner_quotes(text)):
            try:
                return json.loads(cand)
            except Exception:
                pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            for cand in (m.group(0), _escape_inner_quotes(m.group(0))):
                try:
                    return json.loads(cand)
                except Exception:
                    pass
        repaired = _repair_truncated_json(text)
        if repaired:
            try:
                return json.loads(_escape_inner_quotes(repaired))
            except Exception:
                pass
        logger.warning(f"ReflectionAgent JSON 解析失败：{text[:200]}")
        return {}

    @staticmethod
    def _format_notes(result: dict) -> list[str]:
        notes = []
        for grp in result.get("duplicates", []):
            if isinstance(grp, list) and len(grp) >= 2:
                notes.append(f"重复：{', '.join(grp)} 描述高度重叠，建议合并或去重")
        for inc in result.get("severity_inconsistent", []):
            if isinstance(inc, dict):
                ids = inc.get("risk_ids", [])
                notes.append(f"严重度不一致：{', '.join(ids)} — {inc.get('note','')}")
        for miss in result.get("missing", []):
            if isinstance(miss, dict):
                notes.append(f"可能遗漏：{miss.get('topic','')} — {miss.get('reason','')}")
        return notes
