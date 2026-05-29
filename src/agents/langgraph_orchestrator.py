"""LangGraph 编排器。

为什么用 LangGraph：
- 内置流程图可视化（mermaid）
- 内置状态持久化 / 中断恢复
- 内置条件路由（阶段 3 PlannerAgent / 阶段 5 ReflectionAgent 用得上）

为什么仍保留自研 Agent 类：
- LangGraph 节点签名为 (state) -> dict[partial-state]，自研 Agent 直接修改 state
  对象，更直观，face-面试时也能讲清"我没被框架绑死，节点逻辑都是自己写的"
- LangGraph 仅作为编排层
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, END

from .audit_agent import AuditAgent
from .base import BaseAgent
from .indexer_agent import IndexerAgent
from .parser_agent import ParserAgent
from .qa_agent import QAAgent
from .state import SharedState

logger = logging.getLogger(__name__)


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "parser": ParserAgent,
    "indexer": IndexerAgent,
    "qa": QAAgent,
    "audit": AuditAgent,
}


def _node_factory(agent: BaseAgent):
    """LangGraph 节点：调用 agent.invoke 后返回完整 state。"""
    def node(state: SharedState) -> SharedState:
        return agent.invoke(state)
    node.__name__ = f"node_{agent.name}"
    return node


class LangGraphOrchestrator:
    """LangGraph StateGraph 版编排器。

    plan 决定要哪些 agent + 顺序：默认 parser → indexer → qa → audit。
    阶段 3 起，plan 由 PlannerAgent 在 graph 内动态生成。
    """

    DEFAULT_PLAN = ["parser", "indexer", "qa", "audit"]

    def __init__(self, plan: Optional[list[str]] = None):
        self.plan = plan or self.DEFAULT_PLAN
        self._validate_plan(self.plan)
        self.graph = self._build_graph(self.plan)

    @staticmethod
    def _validate_plan(plan: list[str]):
        for name in plan:
            if name not in AGENT_REGISTRY:
                raise ValueError(f"未知 agent: {name}")

    def _build_graph(self, plan: list[str]):
        builder = StateGraph(SharedState)

        # 添加节点
        agent_instances = []
        for name in plan:
            agent = AGENT_REGISTRY[name]()
            builder.add_node(name, _node_factory(agent))
            agent_instances.append(agent)

        # 线性边：plan[0] -> plan[1] -> ... -> END
        builder.set_entry_point(plan[0])
        for i in range(len(plan) - 1):
            builder.add_edge(plan[i], plan[i + 1])
        builder.add_edge(plan[-1], END)

        return builder.compile()

    def run(self, state: SharedState) -> SharedState:
        logger.info(f"LangGraph 启动，plan: {self.plan}")
        # LangGraph 0.6+ 接受 dataclass / TypedDict / dict
        result = self.graph.invoke(state)

        # invoke 返回的可能是 SharedState 或 dict（取决于版本）
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
        """把流程图导出为 mermaid 文本，用于 README。"""
        try:
            mermaid = self.graph.get_graph().draw_mermaid()
            out = Path("docs/agent_workflow.mmd")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(mermaid, encoding="utf-8")
            logger.info(f"mermaid 流程图 → {out}")
        except Exception as e:
            logger.warning(f"导出 mermaid 失败: {e}")
