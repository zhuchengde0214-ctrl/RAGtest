"""DiffAgent：跨合同条款级对比。

核心算法：
1. 把 v1 / v2 各自按 section_path 聚合为"章节体"（每章节合并所有 paragraph/list/table 内容）
2. 用 section_path 做粗对齐（精确匹配 + 后缀匹配 fallback）
3. 同名章节用 LLM 比对，输出 changed / added / removed / unchanged 四类条款级 diff
4. v1 独有章节 → removed_section；v2 独有 → added_section
5. 输出 outputs/diff_results.json，每条 diff 含证据回链（v1/v2 chunk_id + quote）

依赖：state.parsed_doc + state.parsed_doc_v2 + state.retriever + state.retriever_v2
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from llm_client import get_default_model, make_llm_client
from qa_engine import _escape_inner_quotes, _repair_truncated_json

from .base import BaseAgent
from .state import SharedState

logger = logging.getLogger(__name__)


DIFF_PROMPT = """你是合同审查律师。下面是同一份合同的同一章节的两个版本，请对比 v1 与 v2 在该章节的差异。

## 章节：{section}

### v1（旧版）
{v1_text}

### v2（新版）
{v2_text}

## 输出 JSON 数组（不要 ```），每条 diff 一个对象：
[
  {{
    "diff_type": "changed | added | removed",
    "topic": "≤25 字主题，例如『合同总价』『第二期付款比例』",
    "v1_quote": "v1 原文短引用（≤80 字）；diff_type=added 时填 \"\"",
    "v2_quote": "v2 原文短引用（≤80 字）；diff_type=removed 时填 \"\"",
    "summary": "≤80 字描述变化",
    "impact": "low | medium | high",
    "needs_human_review": true|false
  }}
]

要求：
- 只输出真实的差异；如果 v1 和 v2 在本章节实质相同（仅排版/标点差异），输出空数组 []
- impact 判断标准：金额/付款节点/违约责任/工期等关键条款 = high；非核心条款 = medium；纯措辞调整 = low
- diff_type=added：v2 有但 v1 没有的条款
- diff_type=removed：v1 有但 v2 没有的条款
- diff_type=changed：两版都有但内容变了"""


class DiffAgent(BaseAgent):
    name = "diff"

    def __init__(self):
        super().__init__()
        self.client = make_llm_client()
        self.model = get_default_model()

    def check_preconditions(self, state):
        if state.parsed_doc is None:
            return "需要先跑 parser（v1）"
        if state.parsed_doc_v2 is None:
            return "需要先跑 parser 处理 v2（pdf_path_v2 未提供？）"
        return None

    def _run(self, state: SharedState) -> None:
        v1_sections = self._group_by_section(state.parsed_doc.blocks)
        v2_sections = self._group_by_section(state.parsed_doc_v2.blocks)

        state.log(self.name, f"v1 章节 {len(v1_sections)} / v2 章节 {len(v2_sections)}")

        # 对齐
        v1_keys = set(v1_sections)
        v2_keys = set(v2_sections)
        common = v1_keys & v2_keys
        only_v1 = v1_keys - v2_keys
        only_v2 = v2_keys - v1_keys

        # 模糊对齐：only_v1 / only_v2 间用 section_path 后缀做匹配
        matched_pairs, unmatched_v1, unmatched_v2 = self._fuzzy_align(only_v1, only_v2)

        diffs: list[dict] = []
        diff_id = 1

        # 1) 同名章节直接比对
        for sec in sorted(common):
            section_diffs = self._diff_section(sec, v1_sections[sec], v2_sections[sec])
            for d in section_diffs:
                d["diff_id"] = f"D{diff_id:03d}"
                d["section"] = sec
                diff_id += 1
                diffs.append(d)

        # 2) 模糊对齐章节
        for v1_sec, v2_sec in matched_pairs:
            section_diffs = self._diff_section(
                f"{v1_sec} ↔ {v2_sec}", v1_sections[v1_sec], v2_sections[v2_sec]
            )
            for d in section_diffs:
                d["diff_id"] = f"D{diff_id:03d}"
                d["section"] = v1_sec
                d["v2_section"] = v2_sec
                diff_id += 1
                diffs.append(d)

        # 3) 仅 v1 有的章节 → removed_section
        for sec in sorted(unmatched_v1):
            text = v1_sections[sec][:300]
            diffs.append({
                "diff_id": f"D{diff_id:03d}",
                "section": sec,
                "diff_type": "removed_section",
                "topic": f"v2 删除章节：{sec[:30]}",
                "v1_quote": text,
                "v2_quote": "",
                "summary": f"v1 中存在的章节『{sec}』在 v2 中被删除或重组",
                "impact": "high",
                "needs_human_review": True,
            })
            diff_id += 1

        # 4) 仅 v2 有的章节 → added_section
        for sec in sorted(unmatched_v2):
            text = v2_sections[sec][:300]
            diffs.append({
                "diff_id": f"D{diff_id:03d}",
                "section": sec,
                "diff_type": "added_section",
                "topic": f"v2 新增章节：{sec[:30]}",
                "v1_quote": "",
                "v2_quote": text,
                "summary": f"v2 新增章节『{sec}』",
                "impact": "high",
                "needs_human_review": True,
            })
            diff_id += 1

        # 用 retriever 回链每条 diff 的证据 chunk
        diffs = self._enrich_with_chunks(diffs, state)

        state.diff_results = diffs

        # 落盘
        out = self._output_path(state)
        out.write_text(
            json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state.log(
            self.name,
            "diff 完成",
            n_diffs=len(diffs),
            n_changed=sum(1 for d in diffs if d.get("diff_type") == "changed"),
            n_added=sum(1 for d in diffs if d.get("diff_type") == "added"),
            n_removed=sum(1 for d in diffs if d.get("diff_type") == "removed"),
            n_section_added=sum(1 for d in diffs if d.get("diff_type") == "added_section"),
            n_section_removed=sum(1 for d in diffs if d.get("diff_type") == "removed_section"),
        )

    @staticmethod
    def _output_path(state):
        if state.contract_id:
            from contract_library import ContractLibrary
            lib = ContractLibrary()
            paths = lib.paths(state.contract_id)
            paths["base"].mkdir(parents=True, exist_ok=True)
            return paths["diff_results"]
        return Path(state.output_dir) / "diff_results.json"

    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_section(blocks) -> dict[str, str]:
        """同 section_path 的 block 内容拼接成大段落，用于 LLM 比对。"""
        bag: dict[str, list[str]] = defaultdict(list)
        for b in blocks:
            sec = b.section_path or "(根级别)"
            content = b.content.strip()
            if content:
                bag[sec].append(content)
        return {k: "\n".join(v) for k, v in bag.items()}

    @staticmethod
    def _fuzzy_align(only_v1: set[str], only_v2: set[str]) -> tuple[list, set, set]:
        """对两侧独有章节做后缀模糊匹配。
        例如 v1 "1.6 付款方式 > 第三期" 和 v2 "1.6 付款" 可能是同一段。
        """
        if not only_v1 or not only_v2:
            return [], only_v1, only_v2
        matched = []
        used_v2 = set()
        v1_remain = set(only_v1)

        v1_list = list(only_v1)
        v2_list = list(only_v2)

        for v1_sec in v1_list:
            v1_tail = v1_sec.split(">")[-1].strip()
            for v2_sec in v2_list:
                if v2_sec in used_v2:
                    continue
                v2_tail = v2_sec.split(">")[-1].strip()
                # 完全一致或一方包含另一方
                if v1_tail == v2_tail or v1_tail in v2_tail or v2_tail in v1_tail:
                    matched.append((v1_sec, v2_sec))
                    used_v2.add(v2_sec)
                    v1_remain.discard(v1_sec)
                    break
        return matched, v1_remain, only_v2 - used_v2

    # ------------------------------------------------------------------
    def _diff_section(self, section: str, v1_text: str, v2_text: str) -> list[dict]:
        if v1_text.strip() == v2_text.strip():
            return []  # 完全相同

        prompt = DIFF_PROMPT.format(
            section=section,
            v1_text=v1_text[:4000],
            v2_text=v2_text[:4000],
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
        except Exception as e:
            logger.warning(f"DiffAgent LLM 调用失败 section={section}: {e}")
            return [{
                "diff_type": "changed",
                "topic": f"对比失败（{section}）",
                "v1_quote": v1_text[:80],
                "v2_quote": v2_text[:80],
                "summary": f"LLM 调用失败：{e}",
                "impact": "medium",
                "needs_human_review": True,
            }]

        return self._parse_array(raw)

    @staticmethod
    def _parse_array(text: str) -> list:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        def _try(s: str):
            try:
                d = json.loads(s)
                return d if isinstance(d, list) else None
            except Exception:
                return None

        d = _try(text)
        if d is not None:
            return d
        m = re.search(r"\[\s*[\s\S]*\]\s*$", text)
        cand = m.group(0) if m else text
        d = _try(cand) or _try(_escape_inner_quotes(cand))
        if d is not None:
            return d
        repaired = _repair_truncated_json(text)
        if repaired:
            d = _try(repaired) or _try(_escape_inner_quotes(repaired))
            if d is not None:
                return d
        logger.warning(f"DiffAgent JSON 解析失败：{text[:200]}")
        return []

    # ------------------------------------------------------------------
    def _enrich_with_chunks(self, diffs: list[dict], state: SharedState) -> list[dict]:
        """用 retriever.locate_evidence 把 v1_quote/v2_quote 回链到具体 chunk_id。"""
        for d in diffs:
            v1q = d.get("v1_quote") or ""
            v2q = d.get("v2_quote") or ""
            d["v1_evidence"] = self._locate(v1q, state.retriever)
            d["v2_evidence"] = self._locate(v2q, state.retriever_v2)
        return diffs

    @staticmethod
    def _locate(quote: str, retriever) -> dict:
        if not quote or retriever is None:
            return {}
        located = retriever.locate_evidence(quote, top_n=1)
        if not located:
            return {"resolved": False, "matched_quote": quote[:100]}
        ev = located[0]
        return {
            "chunk_id": ev["chunk_id"],
            "section": ev["section"],
            "page_hint": ev["page_hint"],
            "match_type": ev["match_type"],
            "resolved": True,
        }
