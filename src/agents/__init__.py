"""Multi-agent system for contract RAG and audit.

公开接口：
    SharedState        所有 Agent 共享的工作状态
    BaseAgent          Agent 抽象基类
    ParserAgent        PDF 解析（Vision OCR）
    IndexerAgent       分块 + 检索索引
    QAAgent            三类 RAG 问答
    AuditAgent         10 维度合同审查
    PlannerAgent       LLM 决策执行计划（阶段 3）
    DiffAgent          跨合同对比（阶段 4）
    ReflectionAgent    自我审视 + 触发重跑（阶段 5）
    Orchestrator       自研编排器
    LangGraphOrchestrator  LangGraph 编排器（阶段 2）
"""

from .state import SharedState, AgentMessage
from .base import BaseAgent
from .parser_agent import ParserAgent
from .indexer_agent import IndexerAgent
from .qa_agent import QAAgent
from .audit_agent import AuditAgent
from .planner_agent import PlannerAgent
from .diff_agent import DiffAgent
from .reflection_agent import ReflectionAgent
from .intent_router import IntentRouter
from .orchestrator import Orchestrator

__all__ = [
    "SharedState",
    "AgentMessage",
    "BaseAgent",
    "ParserAgent",
    "IndexerAgent",
    "QAAgent",
    "AuditAgent",
    "PlannerAgent",
    "DiffAgent",
    "ReflectionAgent",
    "IntentRouter",
    "Orchestrator",
]
