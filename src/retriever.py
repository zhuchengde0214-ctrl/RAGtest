"""
检索模块

特性：
- Dense: ChromaDB + 可切换 OpenAI / 本地 SentenceTransformer
- Sparse: BM25 (rank_bm25) + jieba 中文分词 + 2-gram 兜底
- Hybrid: Reciprocal Rank Fusion (RRF) + 可选加权分融合
- Rerank: 可选 LLM rerank
- 证据回链: locate_evidence(quote) → 找到原文 chunk
- 多轮: rewrite_query 用 LLM 做指代消解（保留前轮关键实体）

降级路径：
- 本地 embedding 模型不可用 → 自动跳过 dense，纯 BM25
- ChromaDB 持久化失败 → 内存模式
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import jieba
from rank_bm25 import BM25Okapi

from chunker import Chunk

logger = logging.getLogger(__name__)

# 关闭 jieba 启动日志
jieba.setLogLevel(logging.WARNING)


# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------
class EmbeddingProvider:
    """支持 OpenAI / 本地 sentence-transformers / 纯关键词降级"""

    def __init__(
        self,
        use_local: bool = False,
        openai_api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        local_model: str = "all-MiniLM-L6-v2",
    ):
        self.use_local = use_local
        self.disabled = False
        self.model_name = local_model if use_local else model
        self._local = None
        self._openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._openai_model = model
        self._local_model_name = local_model

        if use_local:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"加载本地 Embedding 模型: {local_model}")
                self._local = SentenceTransformer(local_model)
            except Exception as e:
                logger.warning(f"本地 Embedding 加载失败，将禁用 dense 检索: {e}")
                self.disabled = True
        else:
            if not self._openai_key or self._openai_key.startswith("sk-xxxx"):
                logger.warning("未检测到有效 OPENAI_API_KEY，dense 检索禁用")
                self.disabled = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.disabled:
            return []
        if self.use_local:
            return self._local.encode(texts, normalize_embeddings=True).tolist()
        from openai import OpenAI
        client = OpenAI(api_key=self._openai_key)
        # OpenAI 单次最多 ~2048 个输入；这里 batch 已在外部控制
        resp = client.embeddings.create(model=self._openai_model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, query: str) -> list[float]:
        out = self.embed([query])
        return out[0] if out else []


# ------------------------------------------------------------------
# 中文分词
# ------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\s，。！？、；：（）【】《》「」『』""''\.,!?;:\(\)\[\]<>\"'`~@#$%^&*\+\-\=/\\|]")


def tokenize_chinese(text: str) -> list[str]:
    """jieba 分词 + 2-gram 兜底，便于覆盖未登录词"""
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = []
    for seg in jieba.cut_for_search(text):
        seg = seg.strip()
        if not seg or _PUNCT_RE.fullmatch(seg):
            continue
        tokens.append(seg)
    # 2-gram 中文兜底，提高同义/近义召回
    chinese_only = re.sub(r"[^一-鿿]", "", text)
    for i in range(len(chinese_only) - 1):
        tokens.append(chinese_only[i:i + 2])
    return tokens


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------
class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str = "./chroma_db",
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        rerank_top_k: int = 6,
        rrf_k: int = 60,
        collection_name: str = "contract_docs",
    ):
        self.embedding_provider = embedding_provider
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        self.collection_name = collection_name

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as e:
            logger.warning(f"ChromaDB 持久化失败，回退内存模式: {e}")
            self.chroma_client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        self.collection = None
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._chunk_by_id: dict[str, Chunk] = {}
        self._conversation_history: list[dict] = []

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def index(self, chunks: list[Chunk], reuse_existing: bool = False):
        """建立索引。
        reuse_existing=True：如果 chroma collection 已存在且 count 与 chunks 数一致，跳过 dense 重建。
                            BM25 必须每次重建（在内存里，无持久化）。
        """
        self.chunks = chunks
        self._chunk_by_id = {c.chunk_id: c for c in chunks}
        logger.info(f"建立索引（collection={self.collection_name}），{len(chunks)} 个 chunk")

        # BM25 - 必须重建（不持久化）
        tokenized = [tokenize_chinese(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("  BM25 索引完成")

        # Dense
        if self.embedding_provider.disabled:
            logger.warning("  Dense 检索已禁用（embedding 不可用）")
            return

        # 复用已有 collection（长期记忆场景）
        if reuse_existing:
            try:
                existing = self.chroma_client.get_collection(self.collection_name)
                if existing.count() == len(chunks):
                    self.collection = existing
                    logger.info(f"  复用已有 collection（{existing.count()} 个向量）")
                    return
                else:
                    logger.info(
                        f"  collection 已存在但数量不匹配（{existing.count()} vs {len(chunks)}），重建"
                    )
            except Exception:
                logger.info(f"  collection {self.collection_name} 不存在，新建")

        try:
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            batch = 32
            for i in range(0, len(chunks), batch):
                sub = chunks[i:i + batch]
                texts = [c.content for c in sub]
                embs = self.embedding_provider.embed(texts)
                if not embs:
                    raise RuntimeError("embedding 返回空")
                self.collection.add(
                    embeddings=embs,
                    documents=texts,
                    ids=[c.chunk_id for c in sub],
                    metadatas=[self._sanitize_metadata(c.metadata) for c in sub],
                )
                logger.info(f"  Dense 索引: {min(i + batch, len(chunks))}/{len(chunks)}")
        except Exception as e:
            logger.warning(f"Dense 索引失败，回退纯 BM25: {e}")
            self.collection = None

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """ChromaDB 不接受 None / list / dict，转成基本类型"""
        out = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, list):
                out[k] = ",".join(str(x) for x in v)
            elif isinstance(v, dict):
                continue  # 跳过嵌套
            else:
                out[k] = str(v)
        return out

    # ------------------------------------------------------------------
    # 检索（RRF 融合）
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_section: Optional[str] = None,
        filter_block_type: Optional[str] = None,
    ) -> list[dict]:
        top_k = top_k or self.rerank_top_k

        dense_ranking = self._dense_search(query)
        sparse_ranking = self._bm25_search(query)

        # RRF 融合
        scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, cid in enumerate(sparse_ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in ranked:
            c = self._chunk_by_id.get(cid)
            if c is None:
                continue
            if filter_section and filter_section not in c.metadata.get("section_path", ""):
                continue
            if filter_block_type and c.metadata.get("block_type") != filter_block_type:
                continue
            results.append({
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
                "score": score,
                "dense_rank": dense_ranking.index(cid) + 1 if cid in dense_ranking else None,
                "sparse_rank": sparse_ranking.index(cid) + 1 if cid in sparse_ranking else None,
            })
            if len(results) >= top_k:
                break
        return results

    def _dense_search(self, query: str) -> list[str]:
        if self.collection is None or self.embedding_provider.disabled:
            return []
        try:
            qemb = self.embedding_provider.embed_query(query)
            if not qemb:
                return []
            res = self.collection.query(
                query_embeddings=[qemb],
                n_results=self.vector_top_k,
                include=["documents", "distances"],
            )
            return list(res["ids"][0]) if res and res.get("ids") else []
        except Exception as e:
            logger.warning(f"Dense 检索失败: {e}")
            return []

    def _bm25_search(self, query: str) -> list[str]:
        if self.bm25 is None:
            return []
        tokens = tokenize_chinese(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # 仅返回有正分的
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            out.append(self.chunks[i].chunk_id)
            if len(out) >= self.bm25_top_k:
                break
        return out

    # ------------------------------------------------------------------
    # 证据回链
    # ------------------------------------------------------------------
    def locate_evidence(self, quote: str, top_n: int = 1) -> list[dict]:
        """把任意一段文字（来自 LLM 的引用）反查到最相似的 chunk。
        组合策略：子串精确匹配优先，否则用 BM25 排序。"""
        if not quote or not self.chunks:
            return []
        quote = quote.strip()

        # 1) 精确子串匹配（去除空白）
        normalized = re.sub(r"\s+", "", quote)
        for c in self.chunks:
            if normalized and normalized in re.sub(r"\s+", "", c.content):
                return [self._format_evidence(c, match_type="exact_substring", quote=quote)]

        # 2) BM25 粗排
        ranked_ids = self._bm25_search(quote)
        out = []
        for cid in ranked_ids[:top_n]:
            c = self._chunk_by_id.get(cid)
            if c:
                out.append(self._format_evidence(c, match_type="bm25", quote=quote))
        return out

    def _format_evidence(self, chunk: Chunk, match_type: str, quote: str) -> dict:
        meta = chunk.metadata
        return {
            "chunk_id": chunk.chunk_id,
            "section": meta.get("section_path", "") or "(未知章节)",
            "page_hint": meta.get("page_hint"),
            "pages": meta.get("pages"),
            "block_type": meta.get("block_type"),
            "table_id": meta.get("table_id"),
            "match_type": match_type,
            "matched_quote": quote,
            "snippet": chunk.content[:200],
        }

    # ------------------------------------------------------------------
    # LLM rerank
    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: list[dict], llm_client=None, model: Optional[str] = None) -> list[dict]:
        if not candidates or len(candidates) <= 2 or llm_client is None:
            return candidates

        body = ""
        for i, c in enumerate(candidates):
            sec = c["metadata"].get("section_path", "")
            body += f"\n[{i + 1}] 章节: {sec}\n{c['content'][:400]}\n"

        prompt = f"""你是一个相关性评分助手。给定一个问题和若干候选片段，请按相关性从高到低排序。

问题：{query}

候选：{body}

只输出编号序列，逗号分隔，例如：3,1,5,2,4。不要其他内容。"""
        try:
            resp = llm_client.messages.create(
                model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            order_idx = []
            for tok in re.split(r"[,\s，]+", text):
                if tok.isdigit():
                    j = int(tok) - 1
                    if 0 <= j < len(candidates):
                        order_idx.append(j)
            seen = set()
            ordered = []
            for j in order_idx:
                if j not in seen:
                    ordered.append(candidates[j])
                    seen.add(j)
            for j, c in enumerate(candidates):
                if j not in seen:
                    ordered.append(c)
            return ordered
        except Exception as e:
            logger.warning(f"Rerank 失败: {e}")
            return candidates

    # ------------------------------------------------------------------
    # 多轮支持
    # ------------------------------------------------------------------
    def reset_conversation(self):
        self._conversation_history = []

    def add_to_history(self, role: str, content: str):
        self._conversation_history.append({"role": role, "content": content})

    @property
    def conversation_history(self) -> list[dict]:
        return self._conversation_history

    def rewrite_query(self, current_query: str, llm_client=None, model: Optional[str] = None) -> str:
        if not self._conversation_history or llm_client is None:
            return current_query
        history = ""
        for h in self._conversation_history[-6:]:
            history += f"\n{h['role']}: {h['content'][:500]}"

        prompt = f"""你是一个对话改写器。给定一段对话历史和最新一句追问，将追问改写为一个独立、自包含的问题。

对话历史:{history}

最新追问: {current_query}

要求：
1. 把"这些/它们/上述/这个"等指代词替换为前文中出现的具体名词
2. 补全省略的主语/宾语
3. 保留前文已识别出的关键实体名称（如系统模块名、合同条款编号、金额等），便于检索
4. 保持原意和语气

只输出改写后的问题，不加引号。"""
        try:
            resp = llm_client.messages.create(
                model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = resp.content[0].text.strip().strip('"').strip("'")
            logger.info(f"问题改写: {current_query!r} -> {rewritten!r}")
            return rewritten or current_query
        except Exception as e:
            logger.warning(f"问题改写失败: {e}")
            return current_query
