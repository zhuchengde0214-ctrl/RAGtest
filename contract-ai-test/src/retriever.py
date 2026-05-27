"""
检索模块

检索策略：
1. Dense Retrieval: embedding 向量检索（ChromaDB）
2. Sparse Retrieval: BM25 关键词检索（rank_bm25）
3. Hybrid Search: 加权融合 dense + sparse 结果
4. Rerank: 使用 LLM 对候选结果重排序

多轮问答策略：
- 维护对话历史
- 对追问进行问题改写（指代消解）
- 改写后的 query 重新检索
- 引用延续：追踪前轮引用的上下文
"""

import json
import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from chunker import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Embedding 提供者 — 支持 OpenAI API 和本地模型"""

    def __init__(self, use_local: bool = False, openai_api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.use_local = use_local
        self.model_name = model

        if use_local:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载本地 Embedding 模型: {model}")
            self._local_model = SentenceTransformer(model)
        else:
            self._openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            self._openai_model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成 embedding 向量"""
        if self.use_local:
            return self._local_model.encode(texts, normalize_embeddings=True).tolist()
        else:
            from openai import OpenAI
            client = OpenAI(api_key=self._openai_key)
            resp = client.embeddings.create(model=self._openai_model, input=texts)
            return [d.embedding for d in resp.data]

    def embed_query(self, query: str) -> list[float]:
        """生成查询 embedding"""
        return self.embed([query])[0]


class Retriever:
    """混合检索引擎"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str = "../data/chroma_db",
        vector_top_k: int = 8,
        bm25_top_k: int = 8,
        rerank_top_k: int = 5,
        hybrid_weight_vector: float = 0.5,
        hybrid_weight_bm25: float = 0.5,
    ):
        self.embedding_provider = embedding_provider
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.hybrid_weight_vector = hybrid_weight_vector
        self.hybrid_weight_bm25 = hybrid_weight_bm25

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = None
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._chunk_texts: list[str] = []
        self._conversation_history: list[dict] = []

    def index(self, chunks: list[Chunk]):
        """建立索引"""
        self.chunks = chunks
        self._chunk_texts = [c.content for c in chunks]
        logger.info(f"开始索引 {len(chunks)} 个 chunks...")

        # 1. Dense 索引 (ChromaDB)
        collection_name = "contract_docs"
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 批量生成 embedding 并插入
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedding_provider.embed(texts)
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas,
            )
            logger.info(f"  向量索引: {i + len(batch)}/{len(chunks)}")

        # 2. Sparse 索引 (BM25)
        tokenized = [self._tokenize(text) for text in self._chunk_texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"  BM25 索引完成")

    def _tokenize(self, text: str) -> list[str]:
        """中文分词（简易方式：按字符+标点分词）"""
        import re
        # 简易分词：按标点、空格切分，同时保留连续的中文字符
        tokens = []
        for word in re.findall(r'[一-鿿]+|[a-zA-Z0-9]+|[^\s]', text):
            if len(word) > 4 and re.match(r'^[一-鿿]+$', word):
                # 对长中文词做2-gram
                for j in range(0, len(word) - 1):
                    tokens.append(word[j:j + 2])
            tokens.append(word)
        return tokens

    def search(
        self,
        query: str,
        use_hybrid: bool = True,
        filter_section: Optional[str] = None,
    ) -> list[dict]:
        """混合检索"""
        logger.info(f"检索: '{query}'")

        # Dense retrieval
        query_embedding = self.embedding_provider.embed_query(query)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.vector_top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Sparse retrieval (BM25)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:self.bm25_top_k]

        # 融合结果
        if use_hybrid:
            return self._hybrid_fusion(
                dense_results, bm25_top_indices, bm25_scores, filter_section
            )
        else:
            return self._format_dense_results(dense_results, filter_section)

    def _hybrid_fusion(
        self,
        dense_results: dict,
        bm25_indices: list[int],
        bm25_scores,
        filter_section: Optional[str],
    ) -> list[dict]:
        """加权融合 dense + sparse 结果"""
        # 归一化分数
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        # 构建融合分数
        scores = {}

        # Dense 分数（distance 越小越好，转换为 similarity）
        for i, (doc_id, distance) in enumerate(zip(
            dense_results["ids"][0],
            dense_results["distances"][0],
        )):
            sim = 1.0 - distance  # cosine distance -> similarity
            scores[doc_id] = scores.get(doc_id, 0) + self.hybrid_weight_vector * sim

        # BM25 分数
        for idx in bm25_indices:
            chunk_id = self.chunks[idx].chunk_id
            norm_score = bm25_scores[idx] / max_bm25
            scores[chunk_id] = scores.get(chunk_id, 0) + self.hybrid_weight_bm25 * norm_score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        for chunk_id, score in ranked[:self.rerank_top_k]:
            for chunk in self.chunks:
                if chunk.chunk_id == chunk_id:
                    if filter_section and filter_section not in chunk.metadata.get("section_path", ""):
                        continue
                    results.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "score": score,
                    })
                    break

        return results

    def _format_dense_results(self, dense_results: dict, filter_section: Optional[str]) -> list[dict]:
        """格式化纯向量检索结果"""
        results = []
        for i, (doc_id, distance) in enumerate(zip(
            dense_results["ids"][0],
            dense_results["distances"][0],
        )):
            sim = 1.0 - distance
            if filter_section:
                meta = dense_results["metadatas"][0][i]
                if filter_section not in meta.get("section_path", ""):
                    continue
            results.append({
                "chunk_id": doc_id,
                "content": dense_results["documents"][0][i],
                "metadata": dense_results["metadatas"][0][i],
                "score": sim,
            })
        return results[:self.rerank_top_k]

    def rerank(self, query: str, candidates: list[dict], llm_client=None) -> list[dict]:
        """使用 LLM 对候选结果重排序

        如果 llm_client 为 None，跳过 rerank，直接按原始分数排序返回。
        """
        if not candidates or llm_client is None:
            return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        if len(candidates) <= 2:
            return candidates

        # 构建 rerank prompt
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"\n--- 文档片段 {i + 1} ---\n"
            candidates_text += f"来源: {c['metadata'].get('section_path', '未知')}\n"
            candidates_text += f"内容: {c['content'][:500]}\n"

        prompt = f"""请评估以下文档片段与查询问题的相关性，按相关性从高到低排序，返回排序后的编号列表。

查询问题: {query}

{candidates_text}

请只输出排序后的编号（如: 3, 1, 5, 2, 4），不要输出其他内容。"""

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            response = message.content[0].text.strip()

            # 解析排序结果
            order = []
            for part in response.split(','):
                try:
                    order.append(int(part.strip()) - 1)
                except ValueError:
                    pass

            reranked = [candidates[i] for i in order if 0 <= i < len(candidates)]
            return reranked if reranked else candidates
        except Exception as e:
            logger.warning(f"Rerank 失败，使用原始排序: {e}")
            return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

    # ---- 多轮问答支持 ----

    def reset_conversation(self):
        """重置对话历史"""
        self._conversation_history = []

    def add_to_history(self, role: str, content: str):
        """添加对话记录"""
        self._conversation_history.append({"role": role, "content": content})

    def rewrite_query(self, current_query: str, llm_client=None) -> str:
        """多轮问题改写 — 消解指代、补全上下文"""
        if not self._conversation_history:
            return current_query

        history_text = ""
        for h in self._conversation_history[-4:]:  # 最近 4 轮
            history_text += f"\n{h['role']}: {h['content']}"

        prompt = f"""以下是一段对话历史，然后是一个追问。请将追问改写为一个独立的、不需要依赖对话上下文就能理解的完整问题。

对话历史:{history_text}

追问: {current_query}

要求：
1. 将指代词（"这些"、"它们"、"上述"等）替换为具体对象
2. 补全省略的主语、宾语
3. 保持原意不变

请只输出改写后的问题，不要加引号或其他标记。"""

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = message.content[0].text.strip()
            logger.info(f"问题改写: '{current_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"问题改写失败，使用原问题: {e}")
            return current_query
