"""Multi-agent 共享状态。

所有 Agent 都接收一个 SharedState 实例，读其中需要的字段，写自己的产物。
LangGraph 也直接以 SharedState 作为 graph 的 state schema。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class AgentMessage:
    """Agent 间的事件 / 日志消息"""
    timestamp: str
    agent: str
    level: str   # info | warning | error
    msg: str
    payload: dict = field(default_factory=dict)


@dataclass
class SharedState:
    """所有 Agent 共享的工作区。
    每个 Agent 读它需要的输入，写自己的输出，状态在 graph 中流动。
    """

    # ---------------- 输入 ----------------
    pdf_path: str = ""
    pdf_path_v2: Optional[str] = None      # 阶段 4 用：第二份合同（diff 对象）
    user_request: str = ""                 # 阶段 3 用：自然语言需求
    output_dir: str = "outputs"
    cache_dir: Optional[str] = None

    # ---------------- 阶段 3 PlannerAgent 产物 ----------------
    plan: list[str] = field(default_factory=list)       # ["parser","indexer","qa","audit"]
    plan_reasoning: str = ""

    # ---------------- 阶段 1 ParserAgent 产物 ----------------
    parsed_doc: Any = None                 # ParsedDocument
    parsed_doc_v2: Any = None              # 阶段 4 用

    # ---------------- 阶段 1 IndexerAgent 产物 ----------------
    chunks: list = field(default_factory=list)
    chunks_v2: list = field(default_factory=list)
    retriever: Any = None
    retriever_v2: Any = None

    # ---------------- 阶段 1 QAAgent 产物 ----------------
    qa_results: list = field(default_factory=list)

    # ---------------- 阶段 1 AuditAgent 产物 ----------------
    risks: list = field(default_factory=list)

    # ---------------- 阶段 4 DiffAgent 产物 ----------------
    diff_results: list = field(default_factory=list)

    # ---------------- 阶段 5 ReflectionAgent 产物 ----------------
    reflection_notes: list = field(default_factory=list)
    reflection_iters: int = 0
    needs_rerun: Optional[str] = None      # 哪个 agent 需要重跑；None 表示完成

    # ---------------- 全局 ----------------
    messages: list[AgentMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ---------------- 工具方法 ----------------
    def log(self, agent: str, msg: str, level: str = "info", **payload):
        self.messages.append(AgentMessage(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            agent=agent,
            level=level,
            msg=msg,
            payload=payload,
        ))

    def has_error(self) -> bool:
        return bool(self.errors)
