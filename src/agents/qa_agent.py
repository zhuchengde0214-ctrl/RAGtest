"""QAAgent：跑 Q1 / Q2 / Q3 三类问答。

输入：state.retriever
输出：state.qa_results（list[dict]，含 5 条记录）+ outputs/qa_results.json
"""

import json
from pathlib import Path

from llm_client import get_default_model
from qa_engine import QAEngine

from .base import BaseAgent
from .state import SharedState


# 题目固定 3 类问题
Q1 = "本项目包含哪些主要系统模块？请列出模块名称，并说明每个模块的主要用途。"
Q2 = [
    "本项目需要交付哪些主要成果物？",
    "这些成果物分别对应哪些验收材料？",
    "如果验收材料缺失，可能影响哪些付款节点？请说明依据。",
]
Q3 = "请综合判断本项目的付款安排、验收条件、交付计划和附件说明之间是否存在冲突或不一致。请逐条说明依据，并指出哪些问题需要人工复核。"


class QAAgent(BaseAgent):
    name = "qa"

    def check_preconditions(self, state):
        if state.retriever is None:
            return "需要先跑 indexer"
        return None

    def _run(self, state: SharedState) -> None:
        engine = QAEngine(model=get_default_model())
        results = []

        # Q1 simple
        results.append(engine.answer_simple(Q1, state.retriever))

        # Q2 multi-turn
        results.extend(engine.answer_multi_turn(Q2, state.retriever))

        # Q3 complex
        results.append(engine.answer_complex(Q3, state.retriever))

        state.qa_results = results
        out_path = Path(state.output_dir) / "qa_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        n_resolved = sum(
            1
            for r in results
            for c in r.get("citations", [])
            if c.get("resolved")
        )
        state.log(
            self.name,
            "QA 完成",
            n_questions=len(results),
            n_resolved_citations=n_resolved,
        )
