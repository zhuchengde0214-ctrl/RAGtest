"""IndexerAgent：分块 + 建索引（dense + BM25）。

输入：state.parsed_doc / state.parsed_doc_v2
输出：state.chunks / state.chunks_v2 + state.retriever / state.retriever_v2
"""

import json
import os
from pathlib import Path

from chunker import DocumentChunker
from retriever import EmbeddingProvider, Retriever

from .base import BaseAgent
from .state import SharedState


class IndexerAgent(BaseAgent):
    name = "indexer"

    def check_preconditions(self, state):
        if state.parsed_doc is None:
            return "需要先跑 parser"
        return None

    def _run(self, state: SharedState) -> None:
        output_dir = Path(state.output_dir)

        # --- 主合同分块 ---
        chunker = DocumentChunker()
        state.chunks = chunker.chunk(state.parsed_doc)
        chunker.save_chunks(state.chunks, str(output_dir / "chunks.json"))
        state.log(self.name, "主合同分块完成", n_chunks=len(state.chunks))

        # --- v2 分块 ---
        if state.parsed_doc_v2 is not None:
            state.chunks_v2 = chunker.chunk(state.parsed_doc_v2)
            with open(output_dir / "chunks_v2.json", "w", encoding="utf-8") as f:
                json.dump(
                    [{"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
                     for c in state.chunks_v2],
                    f, ensure_ascii=False, indent=2,
                )
            state.log(self.name, "v2 分块完成", n_chunks=len(state.chunks_v2))

        # --- 建索引 ---
        use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        embedding = EmbeddingProvider(
            use_local=use_local,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            local_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        state.retriever = Retriever(
            embedding_provider=embedding,
            persist_dir=str(output_dir / "chroma_db"),
            vector_top_k=int(os.environ.get("VECTOR_TOP_K", "10")),
            bm25_top_k=int(os.environ.get("BM25_TOP_K", "10")),
            rerank_top_k=int(os.environ.get("RERANK_TOP_K", "6")),
        )
        state.retriever.index(state.chunks)
        state.log(self.name, "主合同索引完成")

        if state.chunks_v2:
            state.retriever_v2 = Retriever(
                embedding_provider=embedding,
                persist_dir=str(output_dir / "chroma_db_v2"),
                vector_top_k=int(os.environ.get("VECTOR_TOP_K", "10")),
                bm25_top_k=int(os.environ.get("BM25_TOP_K", "10")),
                rerank_top_k=int(os.environ.get("RERANK_TOP_K", "6")),
            )
            state.retriever_v2.index(state.chunks_v2)
            state.log(self.name, "v2 索引完成")
