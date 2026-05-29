"""自研编排器（不依赖 LangGraph 也能跑）。

阶段 1 用它走通流水线；阶段 2 引入 LangGraphOrchestrator 后两者并存。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .audit_agent import AuditAgent
from .base import BaseAgent
from .indexer_agent import IndexerAgent
from .parser_agent import ParserAgent
from .qa_agent import QAAgent
from .state import SharedState

logger = logging.getLogger(__name__)


# Agent 名称 → 类
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "parser": ParserAgent,
    "indexer": IndexerAgent,
    "qa": QAAgent,
    "audit": AuditAgent,
}


class Orchestrator:
    """按 plan（agent 名称列表）顺序执行。
    若 plan 为空则跑默认全套：parser → indexer → qa → audit。
    """

    DEFAULT_PLAN = ["parser", "indexer", "qa", "audit"]

    def __init__(self, plan: Optional[list[str]] = None):
        self.plan = plan or self.DEFAULT_PLAN
        self._agents = self._build_agents(self.plan)

    @staticmethod
    def _build_agents(plan: list[str]) -> list[BaseAgent]:
        agents = []
        for name in plan:
            cls = AGENT_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"未知 agent: {name}")
            agents.append(cls())
        return agents

    def run(self, state: SharedState) -> SharedState:
        logger.info(f"Orchestrator 启动，计划: {self.plan}")
        for agent in self._agents:
            agent.invoke(state)
            if state.has_error():
                logger.error(f"agent {agent.name} 出错，中止后续 agent")
                break
        self._dump_messages(state)
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
            logger.info(f"agent 消息日志 → {out}")
        except Exception as e:
            logger.warning(f"无法 dump agent_trace: {e}")
