"""
RAG 问答引擎

核心思路：让 LLM 用结构化 JSON 输出答案 + 引用 + 置信度，避免事后用关键词回猜引用。

3 类问答：
- Q1 simple    : 单点检索 + rerank → JSON 答案
- Q2 multi_turn: 第 2 轮起改写 query；维护对话历史；citations 可跨轮
- Q3 complex   : 子问题分解 → 多路检索合并去重 → 输出 conflicts[] 数组，
                  每条 conflict 标 fact / inference / human_review

引用回链：
- LLM 在 JSON 中给出 chunk_id 和原文 quote
- 后处理时用 chunk_id 精确取 metadata（section_path/page/block_type）
- 若 LLM 给的 chunk_id 错误，用 retriever.locate_evidence(quote) 兜底
"""

import json
import logging
import os
import re
from typing import Optional

from llm_client import make_llm_client, get_default_model
from retriever import Retriever

logger = logging.getLogger(__name__)


def _escape_inner_quotes(text: str) -> str:
    """LLM 经常在中文字符串字段里嵌套未转义的双引号，导致 JSON 解析失败。
    扫描每个字符串，把其内部裸露的 `"` 替换为中文引号。判断方式：
    在字符串内部，一个 `"` 后面如果不是 `,` `:` `}` `]` 等 JSON 结构符，
    很可能是误用，替换为 `”`。"""
    out = []
    i = 0
    in_str = False
    esc = False
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # in_str
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            # 看后续非空白字符是否是合法的字符串结束符
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            if j >= n or text[j] in ",:}]\n\r":
                out.append('"')
                in_str = False
                i += 1
            else:
                # 误嵌的引号，替换为中文右引
                out.append("”")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_truncated_json(text: str) -> Optional[str]:
    """LLM 输出常因 max_tokens 截断；尝试通过补全闭合括号修复。
    策略：扫描每个字符，记录"完整对象/数组结束"位置作为 anchor，
    截断到最近一个 anchor，然后补齐外层 ]/}。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
    # 找最外层起点
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None
    s = text[start:]
    # 状态机：记录 (栈深度, 完整对象结束的位置)
    stack: list[str] = []
    in_str = False
    esc = False
    safe_anchors: list[tuple[int, int]] = []  # (after_pos, depth_after) — 在该位置之后是合法位点
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            # 完整对象/数组结束，是一个 safe anchor（位置在 ch 之后）
            safe_anchors.append((i + 1, len(stack)))
    if not stack and not in_str:
        return s  # 已经合法

    if not safe_anchors:
        # 连一个完整对象都没有 → 输出空数组/对象
        return "[]" if s.startswith("[") else "{}"

    # 取最后一个 anchor
    cut_pos, depth_at_cut = safe_anchors[-1]
    truncated = s[:cut_pos]
    # 补齐缺失闭合
    # depth_at_cut 是该 anchor 之后还剩下多少层未关闭
    # 由于 anchor 是某个 } 或 ] 之后，stack 里剩的就是要补的
    # 重新扫一遍 truncated 计算实际还剩的栈
    stack2: list[str] = []
    in_str = False
    esc = False
    for ch in truncated:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack2.append(ch)
        elif ch in "}]" and stack2:
            stack2.pop()
    closing = ""
    for ch in reversed(stack2):
        closing += "}" if ch == "{" else "]"
    return truncated + closing


SYSTEM_PROMPT = """你是一个严谨的合同文档分析助手。基于提供的【已编号文档片段】回答问题。

硬性要求：
1. 答案必须严格基于片段内容，禁止编造或引入外部知识
2. 必须按指定的 JSON 结构输出
3. citations 中的 chunk_id 必须直接复制片段头部的 [chunk_id]
4. citations 中的 quote 必须是片段中的原文（连续字符），不要改写
5. 若片段不足以回答，answer 写明"依据不足"，confidence ≤ 0.3
6. 复杂判断要区分【明确事实 / 合理推断 / 需要人工确认】"""


class QAEngine:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.model = model or get_default_model()
        self.client = make_llm_client(api_key=api_key)

    # ------------------------------------------------------------------
    # Q1 简单
    # ------------------------------------------------------------------
    def answer_simple(self, question: str, retriever: Retriever, top_k: int = 6) -> dict:
        logger.info(f"[Q1] {question}")
        results = retriever.search(question, top_k=top_k * 2)
        results = retriever.rerank(question, results, llm_client=self.client, model=self.model)[:top_k]

        prompt = self._build_prompt_simple(question, results)
        raw = self._chat_json(prompt, max_tokens=3500)
        parsed = self._parse_response(raw, results, retriever)

        return {
            "question_id": "Q1",
            "question_type": "simple",
            "question": question,
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "retrieval_notes": {
                "chunk_strategy": "block 分组（同 section_path 合并段落，表格/图示/签署独立成块），递归切分至 ~1000 字符，重叠 120",
                "metadata_fields": ["section_path", "block_type", "page_hint", "pages", "table_id", "block_ids", "needs_review", "char_len", "source_text"],
                "retrieval_method": f"hybrid: dense top {retriever.vector_top_k} + bm25 top {retriever.bm25_top_k} + RRF 融合 + LLM rerank top {top_k}（jieba+2gram 中文分词）",
                "multi_turn_handling": None,
            },
            "confidence": parsed["confidence"],
        }

    # ------------------------------------------------------------------
    # Q2 多轮
    # ------------------------------------------------------------------
    def answer_multi_turn(self, questions: list[str], retriever: Retriever, top_k: int = 6) -> list[dict]:
        logger.info(f"[Q2] 多轮共 {len(questions)} 轮")
        retriever.reset_conversation()
        out: list[dict] = []

        for i, q in enumerate(questions):
            logger.info(f"  -- 第 {i + 1} 轮: {q}")
            rewritten = q if i == 0 else retriever.rewrite_query(q, llm_client=self.client, model=self.model)

            results = retriever.search(rewritten, top_k=top_k * 2)
            results = retriever.rerank(rewritten, results, llm_client=self.client, model=self.model)[:top_k]

            prompt = self._build_prompt_multi_turn(q, rewritten, results, retriever.conversation_history)
            raw = self._chat_json(prompt, max_tokens=4000)
            parsed = self._parse_response(raw, results, retriever)

            retriever.add_to_history("user", q)
            retriever.add_to_history("assistant", parsed["answer"])

            out.append({
                "question_id": f"Q2-{i + 1}",
                "question_type": "multi_turn",
                "turn": i + 1,
                "question": q,
                "rewritten_question": rewritten if rewritten != q else None,
                "answer": parsed["answer"],
                "citations": parsed["citations"],
                "retrieval_notes": {
                    "chunk_strategy": "同 Q1",
                    "metadata_fields": ["section_path", "block_type", "page_hint", "pages", "table_id", "block_ids"],
                    "retrieval_method": "hybrid (RRF) + LLM rerank",
                    "multi_turn_handling": (
                        "第1轮直接检索；第2轮起用 LLM 改写：保留前轮答案中的关键实体（系统模块名、付款节点、附件编号等）"
                        "并以改写后 query 重检索；对话历史进入 LLM 上下文；引用可来自前轮 chunk"
                    ),
                },
                "confidence": parsed["confidence"],
            })
        return out

    # ------------------------------------------------------------------
    # Q3 复杂
    # ------------------------------------------------------------------
    def answer_complex(self, question: str, retriever: Retriever, per_sub_top_k: int = 5, final_top_k: int = 12) -> dict:
        logger.info(f"[Q3] {question}")
        sub_queries = self._decompose(question)
        logger.info(f"  子问题 {len(sub_queries)} 条: {sub_queries}")

        # 多路检索 + 合并去重
        merged: dict[str, dict] = {}
        for sq in sub_queries:
            res = retriever.search(sq, top_k=per_sub_top_k)
            for r in res:
                cid = r["chunk_id"]
                if cid not in merged:
                    merged[cid] = {**r, "matched_subqueries": [sq]}
                else:
                    merged[cid]["matched_subqueries"].append(sq)
                    merged[cid]["score"] += r["score"]

        candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:final_top_k]
        candidates = retriever.rerank(question, candidates, llm_client=self.client, model=self.model)

        prompt = self._build_prompt_complex(question, candidates)
        raw = self._chat_json(prompt, max_tokens=8000)
        parsed = self._parse_complex_response(raw, candidates, retriever)

        return {
            "question_id": "Q3",
            "question_type": "complex_reasoning",
            "question": question,
            "answer": parsed["answer"],
            "conflicts": parsed["conflicts"],
            "citations": parsed["citations"],
            "retrieval_notes": {
                "chunk_strategy": "同 Q1",
                "metadata_fields": ["section_path", "block_type", "page_hint", "pages", "table_id", "block_ids", "needs_review", "char_len", "source_text"],
                "retrieval_method": (
                    f"问题分解 → {len(sub_queries)} 个子问题分别 hybrid 检索 (RRF) → "
                    f"按 chunk_id 合并去重并叠加 RRF 分 → top {final_top_k} → LLM rerank"
                ),
                "multi_turn_handling": None,
                "sub_queries": sub_queries,
            },
            "confidence": parsed["confidence"],
        }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _format_chunks(self, chunks: list[dict]) -> str:
        out = []
        for c in chunks:
            m = c.get("metadata", {})
            head = (
                f"[{c['chunk_id']}] section={m.get('section_path', '') or '?'}"
                f" | type={m.get('block_type', '')}"
                f" | pages={m.get('pages', m.get('page_hint'))}"
            )
            if m.get("table_id"):
                head += f" | table_id={m['table_id']}"
            if m.get("needs_review"):
                head += " | needs_review=true"
            out.append(f"{head}\n{c['content']}")
        return "\n\n---\n\n".join(out)

    def _build_prompt_simple(self, question: str, chunks: list[dict]) -> str:
        return f"""{SYSTEM_PROMPT}

## 已编号文档片段
{self._format_chunks(chunks)}

## 问题
{question}

## 输出 JSON（只输出 JSON 本身，不要 ```）
{{
  "answer": "回答正文（≤500 字，结构化条理化）",
  "citations": [
    {{
      "chunk_id": "片段头部方括号中的 chunk_id（必须从上面复制）",
      "quote": "片段中的原文短引用（≤120 字，连续字符）",
      "section": "片段头部 section= 后的内容",
      "reason": "该引用如何支持回答（≤60 字）"
    }}
  ],
  "confidence": 0.0~1.0
}}"""

    def _build_prompt_multi_turn(self, question: str, rewritten: str, chunks: list[dict], history: list[dict]) -> str:
        history_text = "\n".join(f"{h['role']}: {h['content'][:400]}" for h in history) if history else "(无)"
        return f"""{SYSTEM_PROMPT}

## 对话历史
{history_text}

## 已编号文档片段（用于本轮）
{self._format_chunks(chunks)}

## 当前轮问题
原始: {question}
改写后: {rewritten}

请基于"对话历史"维持指代和上下文，结合"文档片段"作答。

## 输出 JSON
{{
  "answer": "...",
  "citations": [
    {{"chunk_id": "...", "quote": "...", "section": "...", "reason": "..."}}
  ],
  "confidence": 0.0~1.0
}}"""

    def _build_prompt_complex(self, question: str, chunks: list[dict]) -> str:
        return f"""{SYSTEM_PROMPT}

你正在处理一个跨章节综合判断问题。需要：
- 先分别梳理付款 / 验收 / 交付 / 附件的事实
- 再两两/多边对比，找出冲突或不一致
- 标注每条结论的类型：
   * fact            ← 有充分文档证据
   * inference       ← 基于多处细节的逻辑推断
   * human_review    ← 信息不足或存在歧义，需人工复核

## 已编号文档片段
{self._format_chunks(chunks)}

## 问题
{question}

## 输出 JSON
{{
  "answer": "对整体一致性的总结性判断（≤300 字）",
  "conflicts": [
    {{
      "topic": "冲突主题，例如 付款节点与验收材料",
      "conclusion_class": "fact | inference | human_review",
      "description": "冲突或不一致的具体表述（≤200 字）",
      "evidence": [
        {{
          "chunk_id": "必须从片段头部复制",
          "quote": "原文短引用（≤120 字）",
          "section": "片段头部 section= 内容"
        }}
      ],
      "needs_human_review": true|false
    }}
  ],
  "citations": [
    {{"chunk_id": "...", "quote": "...", "section": "...", "reason": "..."}}
  ],
  "confidence": 0.0~1.0
}}

至少给出 3 条 conflicts；如果整体一致，也要列出"已核对一致"的关键点作为 fact。"""

    # ------------------------------------------------------------------
    # 子问题分解
    # ------------------------------------------------------------------
    def _decompose(self, question: str) -> list[str]:
        prompt = f"""把以下复杂问题拆解为 3-5 个独立、可独立检索的子问题。

复杂问题: {question}

每行输出一个子问题，不加编号、不加引号、不加解释。"""
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            subs = [re.sub(r"^[\d\.\)、\-\s]+", "", ln).strip() for ln in text.split("\n") if ln.strip()]
            subs = [s for s in subs if len(s) > 4][:5]
            return subs or [question]
        except Exception as e:
            logger.warning(f"子问题分解失败: {e}")
            return [question]

    # ------------------------------------------------------------------
    # LLM 调用 + JSON 解析
    # ------------------------------------------------------------------
    def _chat_json(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _safe_json_parse(self, text: str) -> Optional[dict]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 1) 直接解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2) 抽取最外层 JSON 对象
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        cand = m.group(0) if m else text
        try:
            return json.loads(cand)
        except Exception:
            pass
        # 3) 转义字符串内部裸双引号
        try:
            return json.loads(_escape_inner_quotes(cand))
        except Exception:
            pass
        # 4) 修复被截断的 JSON
        repaired = _repair_truncated_json(text)
        if repaired is not None:
            try:
                return json.loads(_escape_inner_quotes(repaired))
            except Exception:
                pass
        return None

    def _parse_response(self, raw: str, candidates: list[dict], retriever: Retriever) -> dict:
        data = self._safe_json_parse(raw) or {}
        answer = (data.get("answer") or "").strip() or "(模型输出无法解析为 JSON，原始输出已记录在日志)"
        citations_raw = data.get("citations") or []
        citations = self._normalize_citations(citations_raw, candidates, retriever)
        confidence = float(data.get("confidence", 0.5)) if isinstance(data.get("confidence"), (int, float)) else 0.5
        if not citations and candidates:
            confidence = min(confidence, 0.4)
        return {"answer": answer, "citations": citations, "confidence": max(0.0, min(1.0, confidence))}

    def _parse_complex_response(self, raw: str, candidates: list[dict], retriever: Retriever) -> dict:
        data = self._safe_json_parse(raw) or {}
        answer = (data.get("answer") or "").strip() or "(模型输出无法解析为 JSON)"
        conflicts_raw = data.get("conflicts") or []
        citations_raw = data.get("citations") or []
        confidence = float(data.get("confidence", 0.5)) if isinstance(data.get("confidence"), (int, float)) else 0.5

        conflicts = []
        for cf in conflicts_raw:
            if not isinstance(cf, dict):
                continue
            evidence = self._normalize_citations(cf.get("evidence") or [], candidates, retriever)
            cls = cf.get("conclusion_class", "human_review")
            if cls not in ("fact", "inference", "human_review"):
                cls = "human_review"
            needs_review = cf.get("needs_human_review")
            if not isinstance(needs_review, bool):
                needs_review = cls != "fact"
            conflicts.append({
                "topic": cf.get("topic", ""),
                "conclusion_class": cls,
                "description": cf.get("description", ""),
                "evidence": evidence,
                "needs_human_review": needs_review,
            })

        citations = self._normalize_citations(citations_raw, candidates, retriever)
        if not citations:
            # 用 conflicts 内的 evidence 兜底成 citations
            for cf in conflicts:
                citations.extend(cf.get("evidence", []))

        return {
            "answer": answer,
            "conflicts": conflicts,
            "citations": citations[:20],
            "confidence": max(0.0, min(1.0, confidence)),
        }

    # ------------------------------------------------------------------
    # 引用归一化（关键：把 LLM 给的 chunk_id/quote 回链到真实 chunk）
    # ------------------------------------------------------------------
    def _normalize_citations(self, citations_raw: list, candidates: list[dict], retriever: Retriever) -> list[dict]:
        if not isinstance(citations_raw, list):
            return []
        cand_by_id = {c["chunk_id"]: c for c in candidates}
        out = []
        for cit in citations_raw:
            if not isinstance(cit, dict):
                continue
            cid = cit.get("chunk_id", "").strip()
            quote = (cit.get("quote") or "").strip()

            chunk = cand_by_id.get(cid)
            if chunk is None and quote:
                # 兜底：用 quote 反查
                located = retriever.locate_evidence(quote, top_n=1)
                if located:
                    cid = located[0]["chunk_id"]
                    chunk = cand_by_id.get(cid) or {
                        "chunk_id": cid,
                        "metadata": {
                            "section_path": located[0]["section"],
                            "page_hint": located[0]["page_hint"],
                            "pages": located[0]["pages"],
                            "block_type": located[0]["block_type"],
                            "table_id": located[0]["table_id"],
                        },
                        "content": located[0]["snippet"],
                    }

            if chunk is None:
                # 仍然查不到，保留 LLM 原始内容但标记 unresolved
                out.append({
                    "source_id": cid or "unresolved",
                    "section": cit.get("section", "(未解析)"),
                    "page_hint": None,
                    "block_type": None,
                    "quote": quote[:120],
                    "reason": cit.get("reason", ""),
                    "resolved": False,
                })
                continue

            meta = chunk.get("metadata", {})
            out.append({
                "source_id": chunk["chunk_id"],
                "section": meta.get("section_path", "") or cit.get("section", ""),
                "page_hint": meta.get("page_hint"),
                "pages": meta.get("pages"),
                "block_type": meta.get("block_type"),
                "table_id": meta.get("table_id"),
                "quote": quote[:200],
                "reason": cit.get("reason", ""),
                "resolved": True,
            })
        return out
