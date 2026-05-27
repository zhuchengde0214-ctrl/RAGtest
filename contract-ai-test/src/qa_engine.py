"""
RAG 问答引擎

支持三类问答：
1. Q1: 简单事实问题 — 定位单点事实
2. Q2: 多轮问答 — 维护上下文，依次追问
3. Q3: 全文复杂推理 — 跨章节综合判断

每个回答都包含:
- answer: 回答文本
- citations: 引用来源（章节、原文引用、理由）
- retrieval_notes: 检索策略说明
- confidence: 置信度
"""

import json
import logging
import os
import re
from typing import Optional

from anthropic import Anthropic

from chunker import Chunk
from retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QAEngine:
    """RAG 问答引擎"""

    SYSTEM_PROMPT = """你是一个专业的法律文档分析助手。根据提供的文档片段回答问题。

要求：
1. 回答必须基于提供的文档内容，不能编造
2. 如果文档信息不足以回答，明确说明"依据不足"
3. 引用时标注具体的章节、段落或表格
4. 对于复杂推理问题，要区分"明确事实"、"合理推断"和"需要人工确认"
5. 回答使用中文，简洁专业"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY 未设置")
        self.client = Anthropic(api_key=self.api_key)

    def answer_simple(
        self,
        question: str,
        retriever: Retriever,
        top_k: int = 5,
    ) -> dict:
        """简单事实问答"""
        logger.info(f"简单问答: {question}")

        # 检索
        results = retriever.search(question, use_hybrid=True)
        results = retriever.rerank(question, results, llm_client=self.client)[:top_k]

        # 构建上下文
        context = self._build_context(results)

        # 生成回答
        prompt = f"{self.SYSTEM_PROMPT}\n\n## 文档片段\n{context}\n\n## 问题\n{question}\n\n请回答并给出引用来源。对每条引用，标注章节、原文短引用和理由。"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = response.content[0].text

        # 解析引用
        citations = self._extract_citations(answer_text, results)

        return {
            "question_id": "Q1",
            "question_type": "simple",
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieval_notes": {
                "chunk_strategy": "按标题层级切分，表格独立 chunk，段落合并后按语义边界递归切分",
                "metadata_fields": ["section_path", "page_hint", "block_type", "table_id", "source_text"],
                "retrieval_method": "embedding top_k=8 + BM25 top_k=8，加权融合(0.5/0.5)，LLM rerank 至 top 5",
                "multi_turn_handling": None,
            },
            "confidence": self._estimate_confidence(answer_text, results),
        }

    def answer_multi_turn(
        self,
        questions: list[str],
        retriever: Retriever,
        top_k: int = 5,
    ) -> list[dict]:
        """多轮问答"""
        logger.info(f"多轮问答: {len(questions)} 轮")

        retriever.reset_conversation()
        results_list = []

        for i, question in enumerate(questions):
            logger.info(f"  第 {i + 1} 轮: {question}")

            # 问题改写（第2轮起）
            if i > 0:
                rewritten = retriever.rewrite_query(question, llm_client=self.client)
            else:
                rewritten = question

            # 检索
            results = retriever.search(rewritten, use_hybrid=True)
            results = retriever.rerank(rewritten, results, llm_client=self.client)[:top_k]

            # 构建上下文（含历史问答）
            history_context = self._build_conversation_history(retriever._conversation_history)
            context = self._build_context(results)
            full_context = history_context + "\n\n" + context if history_context else context

            # 生成回答
            prompt = f"""{self.SYSTEM_PROMPT}

## 对话历史
{full_context}

## 当前问题（第{i + 1}轮）
{question}

请结合对话历史和文档片段回答问题。给出引用来源。"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            answer_text = response.content[0].text
            citations = self._extract_citations(answer_text, results)

            # 保存对话历史
            retriever.add_to_history("user", question)
            retriever.add_to_history("assistant", answer_text)

            question_id = f"Q2-{i + 1}"

            results_list.append({
                "question_id": question_id,
                "question_type": "multi_turn",
                "turn": i + 1,
                "question": question,
                "rewritten_question": rewritten if i > 0 else None,
                "answer": answer_text,
                "citations": citations,
                "retrieval_notes": {
                    "chunk_strategy": "按标题层级切分，表格独立 chunk",
                    "metadata_fields": ["section_path", "page_hint", "block_type", "table_id"],
                    "retrieval_method": "embedding top_k=8 + BM25 top_k=8，加权融合，LLM rerank",
                    "multi_turn_handling": f"第{i + 1}轮：维护{len(retriever._conversation_history)}条历史记录，使用 LLM 改写指代消解后检索",
                },
                "confidence": self._estimate_confidence(answer_text, results),
            })

        return results_list

    def answer_complex(
        self,
        question: str,
        retriever: Retriever,
        top_k: int = 10,
    ) -> dict:
        """全文复杂推理问答"""
        logger.info(f"复杂推理: {question}")

        # 多角度检索 — 拆分关键子问题
        sub_queries = self._decompose_question(question)

        all_results = []
        for sub_q in sub_queries:
            results = retriever.search(sub_q, use_hybrid=True)
            results = retriever.rerank(sub_q, results, llm_client=self.client)[:top_k]
            all_results.extend(results)

        # 去重
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                unique_results.append(r)
        unique_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        # 构建上下文
        context = self._build_context(unique_results)

        # 生成回答
        prompt = f"""{self.SYSTEM_PROMPT}

## 综合文档片段
{context}

## 复杂推理问题
{question}

请综合判断并回答。要求：
1. 逐条分析依据，标注引用来源
2. 区分以下三类结论：
   - 【明确事实】有充分文档证据支持的
   - 【合理推断】基于多处细节的逻辑推断
   - 【需要人工确认】文档信息不足或存在歧义，需人工复核的
3. 如果存在冲突或不一致，明确指出冲突点和涉及的具体条款"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = response.content[0].text
        citations = self._extract_citations(answer_text, unique_results)

        return {
            "question_id": "Q3",
            "question_type": "complex_reasoning",
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieval_notes": {
                "chunk_strategy": "按标题层级切分，表格独立 chunk，段落合并后递归切分",
                "metadata_fields": ["section_path", "page_hint", "block_type", "table_id", "source_text"],
                "retrieval_method": f"多角度检索：将问题分解为{len(sub_queries)}个子问题分别检索，embedding top_k=8 + BM25 top_k=8，LLM rerank，合并去重后取 top {top_k}",
                "multi_turn_handling": None,
            },
            "confidence": self._estimate_confidence(answer_text, unique_results),
        }

    def _decompose_question(self, question: str) -> list[str]:
        """将复杂问题拆解为子问题"""
        prompt = f"""请将以下复杂问题拆解为 3-5 个独立的子问题，每个子问题聚焦一个具体方面。

复杂问题: {question}

请每行输出一个子问题，不要编号或其他标记。"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            sub_queries = [line.strip().lstrip('0123456789.-) ') for line in text.split('\n') if line.strip()]
            logger.info(f"问题拆解: {len(sub_queries)} 个子问题")
            return sub_queries if sub_queries else [question]
        except Exception:
            return [question]

    def _build_context(self, results: list[dict]) -> str:
        """构建上下文文本"""
        parts = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            section = meta.get("section_path", "未知章节")
            page = meta.get("page_hint", "未知")
            block_type = meta.get("block_type", "paragraph")

            parts.append(
                f"### 片段 {i + 1} [类型:{block_type} | 章节:{section} | 页码:{page}]\n"
                f"{r['content']}"
            )
        return "\n\n".join(parts)

    def _build_conversation_history(self, history: list[dict]) -> str:
        """构建对话历史文本"""
        if not history:
            return ""
        parts = ["## 之前的对话"]
        for h in history:
            parts.append(f"- **{h['role']}**: {h['content'][:300]}")
        return "\n".join(parts)

    def _extract_citations(self, answer_text: str, results: list[dict]) -> list[dict]:
        """从回答中提取引用信息"""
        citations = []

        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            section = meta.get("section_path", "未知")
            content = r["content"]

            # 检查回答中是否引用了该片段的内容
            # 使用简单关键词匹配
            keywords = self._extract_keywords(content)
            for kw in keywords[:3]:
                if len(kw) > 4 and kw in answer_text:
                    citations.append({
                        "source_id": r["chunk_id"],
                        "section": section,
                        "quote": kw[:120],
                        "reason": f"回答中引用了该片段的内容",
                    })
                    break

        # 若自动提取太少，手动构建
        if len(citations) < min(3, len(results)):
            for r in results:
                meta = r.get("metadata", {})
                citations.append({
                    "source_id": r["chunk_id"],
                    "section": meta.get("section_path", "未知"),
                    "quote": r["content"][:150],
                    "reason": "检索到的相关文档片段",
                })

        return citations[:8]

    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """提取文本关键词"""
        # 按句号、换行等拆分，取较长的句子作为关键引用
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 6]
        return sorted(sentences, key=len, reverse=True)[:top_n]

    def _estimate_confidence(self, answer: str, results: list[dict]) -> float:
        """估算回答置信度"""
        if not results:
            return 0.1
        if "依据不足" in answer or "无法确定" in answer:
            return 0.3
        if len(results) >= 5:
            return 0.85
        if len(results) >= 3:
            return 0.7
        return 0.5
