"""LangGraph 编排器（含 PlannerAgent 条件路由）。

为什么用 LangGraph：
- 内置流程图可视化（mermaid）
- 内置状态持久化 / 中断恢复
- 内置条件路由 → 阶段 3 PlannerAgent 动态 plan，阶段 5 ReflectionAgent 循环都靠它

图结构（启用 planner 时）：
    START → planner → [parser → indexer → qa/audit/diff →] END
    planner 输出 state.plan，条件边按 plan 决定下一步走谁。

混合架构：
- LangGraph 仅做编排
- 每个节点直接调用 BaseAgent.invoke(state)，节点内部逻辑全部自研
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, END

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


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "parser": ParserAgent,
    "indexer": IndexerAgent,
    "qa": QAAgent,
    "audit": AuditAgent,
    "diff": DiffAgent,
    "reflection": ReflectionAgent,
}


def _agent_node(agent: BaseAgent):
    """LangGraph 节点：调用 agent.invoke 后返回 state。"""
    def node(state: SharedState) -> SharedState:
        return agent.invoke(state)
    node.__name__ = f"node_{agent.name}"
    return node


class LangGraphOrchestrator:
    """LangGraph StateGraph 版编排器。

    两种模式：
    - use_planner=False：固定 plan 跑线性图
    - use_planner=True：planner 节点 + 条件边动态路由
    """

    DEFAULT_PLAN = ["parser", "indexer", "qa", "audit"]

    def __init__(
        self,
        plan: Optional[list[str]] = None,
        use_planner: bool = False,
    ):
        self.use_planner = use_planner
        self.plan = plan or (None if use_planner else self.DEFAULT_PLAN)
        if not use_planner:
            self._validate_plan(self.plan)
        self.graph = self._build()

    @staticmethod
    def _validate_plan(plan: list[str]):
        for name in plan:
            if name not in AGENT_REGISTRY:
                raise ValueError(f"未知 agent: {name}")

    # ------------------------------------------------------------------
    # 图构造
    # ------------------------------------------------------------------
    def _build(self):
        if self.use_planner:
            return self._build_with_planner()
        return self._build_linear(self.plan)

    def _build_linear(self, plan: list[str]):
        """无 planner：固定 plan 线性串成图。"""
        builder = StateGraph(SharedState)
        for name in plan:
            agent = AGENT_REGISTRY[name]()
            builder.add_node(name, _agent_node(agent))
        builder.set_entry_point(plan[0])
        for i in range(len(plan) - 1):
            builder.add_edge(plan[i], plan[i + 1])
        builder.add_edge(plan[-1], END)
        return builder.compile()

    def _build_with_planner(self):
        """启用 planner：intent_router → planner → agents（条件路由）。
        intent=off_topic 时直接 END（IntentRouter 已经写了 lite_reply）。
        """
        builder = StateGraph(SharedState)

        # 入口节点
        builder.add_node("intent_router", _agent_node(IntentRouter()))
        builder.add_node("planner", _agent_node(PlannerAgent()))

        # 业务 agent 节点
        for name, cls in AGENT_REGISTRY.items():
            builder.add_node(name, _agent_node(cls()))

        builder.set_entry_point("intent_router")

        # intent_router → planner（contract_related）/ END（off_topic）
        builder.add_conditional_edges(
            "intent_router",
            self._router_after_intent,
            {"planner": "planner", "__end__": END},
        )

        # planner → 第一个 agent（条件路由）
        builder.add_conditional_edges(
            "planner",
            self._router_after_planner,
            {name: name for name in AGENT_REGISTRY} | {"__end__": END},
        )

        # 每个 agent 跑完 → 查 plan 决定下一个
        for name in AGENT_REGISTRY:
            builder.add_conditional_edges(
                name,
                self._make_agent_router(name),
                {n: n for n in AGENT_REGISTRY} | {"__end__": END},
            )

        return builder.compile()

    @staticmethod
    def _router_after_intent(state: SharedState) -> str:
        if state.intent == "off_topic":
            return "__end__"
        return "planner"

    # ------------------------------------------------------------------
    # 路由函数
    # ------------------------------------------------------------------
    @staticmethod
    def _router_after_planner(state: SharedState) -> str:
        plan = state.plan
        if not plan:
            return "__end__"
        first = plan[0]
        if first not in AGENT_REGISTRY:
            return "__end__"
        return first

    @staticmethod
    def _make_agent_router(current_agent: str):
        def router(state: SharedState) -> str:
            plan = state.plan or []
            if state.has_error():
                return "__end__"

            # ReAct：reflection 节点 → 看 needs_rerun 决定回退到 audit
            if current_agent == "reflection" and state.needs_rerun == "audit":
                state.needs_rerun = None
                return "audit"

            try:
                idx = plan.index(current_agent)
            except ValueError:
                return "__end__"
            for nxt in plan[idx + 1:]:
                if nxt in AGENT_REGISTRY:
                    return nxt
            return "__end__"
        return router

    # ------------------------------------------------------------------
    # 执行入口
    # ------------------------------------------------------------------
    def run(self, state: SharedState) -> SharedState:
        logger.info(
            f"LangGraph 启动，{'use_planner=True' if self.use_planner else f'plan={self.plan}'}"
        )
        # LangGraph 0.6+ 默认有递归保护，把限制提高一些以防 reflection 循环
        result = self.graph.invoke(state, config={"recursion_limit": 50})
        if isinstance(result, dict):
            for k, v in result.items():
                if hasattr(state, k):
                    setattr(state, k, v)
        else:
            state = result

        self._dump_messages(state)
        self._dump_mermaid()
        return state

    @staticmethod
    def _dump_messages(state: SharedState) -> None:
        try:
            out = Path(state.output_dir) / "agent_trace.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    [m.__dict__ for m in state.messages],
                    f, ensure_ascii=False, indent=2,
                )
            logger.info(f"agent trace → {out}")
        except Exception as e:
            logger.warning(f"无法 dump agent_trace: {e}")

    def _dump_mermaid(self) -> None:
        # 自动生成的 mermaid 包含所有可能的条件边，比较杂乱；
        # 写入 _generated.mmd，README 引用的是手写的 docs/agent_workflow.mmd
        try:
            mermaid = self.graph.get_graph().draw_mermaid()
            out = Path("docs/agent_workflow_generated.mmd")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(mermaid, encoding="utf-8")
            logger.info(f"自动生成的 LangGraph mermaid → {out}（仅供调试，README 用 agent_workflow.mmd）")
        except Exception as e:
            logger.warning(f"导出 mermaid 失败: {e}")
