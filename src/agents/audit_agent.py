"""AuditAgent：合同 10 维度风险审查。

输入：state.chunks + state.retriever
输出：state.risks + outputs/review_results.json
"""

import json
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
        risks = engine.review(state.chunks, state.retriever)
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
        )
