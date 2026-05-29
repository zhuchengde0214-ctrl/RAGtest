"""Agent 抽象基类。

每个具体 Agent 实现 _run(state) 并通过 invoke(state) 触发；
基类负责通用的日志、计时、异常捕获，让每个 Agent 实现保持纯粹。
"""

import logging
import time
from abc import ABC, abstractmethod

from .state import SharedState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """所有 Agent 的基类。

    子类需要实现：
        name: str            agent 名称
        _run(state) -> None  核心执行逻辑（直接修改 state）

    可选覆盖：
        check_preconditions(state) -> Optional[str]
            返回前置检查失败原因；None 表示可执行
    """

    name: str = ""

    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    # ---------- 子类必实现 ----------
    @abstractmethod
    def _run(self, state: SharedState) -> None: ...

    # ---------- 公共逻辑 ----------
    def check_preconditions(self, state: SharedState) -> "str | None":
        return None

    def invoke(self, state: SharedState) -> SharedState:
        """统一入口：日志 + 计时 + 异常捕获。LangGraph 节点可直接调用此方法。"""
        skip = self.check_preconditions(state)
        if skip:
            state.log(self.name, f"跳过：{skip}", level="info")
            return state

        state.log(self.name, "开始")
        t0 = time.time()
        try:
            self._run(state)
            elapsed = time.time() - t0
            state.log(self.name, f"完成 ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            err = f"{self.name} 失败 ({elapsed:.1f}s): {e}"
            logger.exception(err)
            state.log(self.name, err, level="error")
            state.errors.append(err)
        return state

    # LangGraph 节点签名要求 (state) -> state；invoke 即满足
    def __call__(self, state: SharedState) -> SharedState:
        return self.invoke(state)
