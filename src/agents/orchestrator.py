"""自研编排器（不依赖 LangGraph）。

支持两种模式：
1. 显式 plan：构造时传 plan=["parser","indexer","qa","audit"]
2. LLM-driven：构造时传 use_planner=True，先跑 PlannerAgent 生成 plan
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .audit_agent import AuditAgent
from .base import BaseAgent
from .diff_agent import DiffAgent
from .indexer_agent import IndexerAgent
from .intent_router import IntentRouter
from .parser_agent import ParserAgent
from .planner_agent import PlannerAgent
from .qa_agent import QAAgent
from .reflection_agent import ReflectionAgent
from .state import SharedState

logger = logging.getLogger(__name__)


# Agent 名称 → 类
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "parser": ParserAgent,
    "indexer": IndexerAgent,
    "qa": QAAgent,
    "audit": AuditAgent,
    "diff": DiffAgent,
    "reflection": ReflectionAgent,
}


class Orchestrator:
    """按 plan 顺序执行 agent。
    若 use_planner=True，先跑 PlannerAgent 生成 plan。
    若 plan 为空且未启用 planner，跑默认全套。
    """

    DEFAULT_PLAN = ["parser", "indexer", "qa", "audit"]

    def __init__(
        self,
        plan: Optional[list[str]] = None,
        use_planner: bool = False,
    ):
        self.use_planner = use_planner
        self.plan = plan or (None if use_planner else self.DEFAULT_PLAN)

    def run(self, state: SharedState) -> SharedState:
        # 0. IntentRouter（仅在启用 planner 时；显式 plan 模式跳过 router）
        if self.use_planner and state.user_request.strip():
            IntentRouter().invoke(state)
            if state.intent == "off_topic":
                self._dump_messages(state)
                return state

        # 1. （可选）先跑 PlannerAgent 决定 plan
        if self.use_planner:
            planner = PlannerAgent()
            planner.invoke(state)
            if state.has_error():
                self._dump_messages(state)
                return state
            plan = state.plan or self.DEFAULT_PLAN
        else:
            plan = self.plan or self.DEFAULT_PLAN
            state.plan = plan
            state.plan_reasoning = state.plan_reasoning or "用户显式指定 plan"

        logger.info(f"Orchestrator 计划: {plan}")
        if state.plan_reasoning:
            logger.info(f"  reasoning: {state.plan_reasoning}")

        # 2. 顺序执行 + ReAct 循环（仅 reflection 触发 audit 重跑）
        index = 0
        max_steps = len(plan) * 3   # 兜底防无限循环
        executed = 0
        while index < len(plan) and executed < max_steps:
            name = plan[index]
            cls = AGENT_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"未知 agent: {name}，跳过")
                index += 1
                continue
            agent = cls()
            agent.invoke(state)
            executed += 1

            if state.has_error():
                logger.error(f"agent {agent.name} 出错，中止后续 agent")
                break

            # ReAct：如果是 reflection 且建议重跑 audit，则回退到 audit
            if name == "reflection" and state.needs_rerun == "audit":
                # 找到上一次 audit 的位置，跳回去
                try:
                    audit_idx = plan.index("audit")
                    state.log("orchestrator", f"ReAct: reflection 触发 audit 重跑（iter={state.reflection_iters}）")
                    state.needs_rerun = None
                    index = audit_idx
                    continue
                except ValueError:
                    pass

            index += 1

        self._dump_messages(state)
        return state

    @staticmethod
    def _build_agents(plan: list[str]) -> list[BaseAgent]:
        agents = []
        for name in plan:
            cls = AGENT_REGISTRY.get(name)
            if cls is None:
                # 阶段 4 的 diff agent 在这里加；现在不在 registry 里就跳过 + warning
                logger.warning(f"未知 agent: {name}（可能是后续阶段才实现），跳过")
                continue
            agents.append(cls())
        return agents

    @staticmethod
    def _dump_messages(state: SharedState) -> None:
        try:
            out = Path(state.output_dir) / "agent_trace.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    [m.__dict__ for m in state.messages],
                    f, ensure_ascii=False, indent=2,
                )
            logger.info(f"agent 消息日志 → {out}")
        except Exception as e:
            logger.warning(f"无法 dump agent_trace: {e}")
