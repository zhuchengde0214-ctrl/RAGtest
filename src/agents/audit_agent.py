"""AuditAgent：合同 10 维度风险审查。

输入：state.chunks + state.retriever
输出：state.risks + outputs/review_results.json

ReAct 支持：若 state.reflection_iters > 0 且 state.reflection_notes 非空，
本次审查会把 reflection notes 作为额外提示传给 ReviewEngine（让 LLM 重点关注遗漏点）。
"""

import json
import os
from pathlib import Path

from llm_client import get_default_model
from review_engine import ReviewEngine

from .base import BaseAgent
from .state import SharedState


class AuditAgent(BaseAgent):
    name = "audit"

    def check_preconditions(self, state):
        if state.retriever is None:
            return "需要先跑 indexer"
        if not state.chunks:
            return "chunks 为空"
        return None

    def _run(self, state: SharedState) -> None:
        engine = ReviewEngine(model=get_default_model())

        # Reflection 之后的重跑：把 reflection_notes 传给 review_engine
        # ReviewEngine 当前 API 不支持额外 hint，最简方式是临时注入到环境变量
        # （review_engine 不需要改造，只是它的 prompt 会读到 hint）。
        prev_hint = os.environ.get("REVIEW_REFLECTION_HINT")
        try:
            if state.reflection_iters > 0 and state.reflection_notes:
                hint = "## 上一轮审视发现的问题（请本轮重点关注）\n" + "\n".join(
                    f"- {n}" for n in state.reflection_notes[-10:]
                )
                os.environ["REVIEW_REFLECTION_HINT"] = hint
                state.log(self.name, f"使用 reflection hint（{len(state.reflection_notes)} 条）")

            risks = engine.review(state.chunks, state.retriever)
        finally:
            # 恢复
            if prev_hint is None:
                os.environ.pop("REVIEW_REFLECTION_HINT", None)
            else:
                os.environ["REVIEW_REFLECTION_HINT"] = prev_hint

        state.risks = risks

        out_path = Path(state.output_dir) / "review_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(risks, f, ensure_ascii=False, indent=2)

        sev_count = {"high": 0, "medium": 0, "low": 0}
        for r in risks:
            sev = r.get("severity", "medium")
            sev_count[sev] = sev_count.get(sev, 0) + 1
        state.log(
            self.name,
            "审查完成",
            n_risks=len(risks),
            severity=sev_count,
            iter=state.reflection_iters,
        )
