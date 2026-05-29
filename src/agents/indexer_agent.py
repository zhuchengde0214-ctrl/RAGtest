"""IndexerAgent：分块 + 建索引（dense + BM25）。

支持 ContractLibrary 长期记忆：
- 每份合同独立 chroma collection（命名 contract_<id>）
- 复用已有 collection（避免重新 embed）
- chunks.json 落到 outputs/contracts/<id>/chunks.json
"""

import json
import os
from pathlib import Path

from chunker import DocumentChunker
from contract_library import ContractLibrary, DEFAULT_CHROMA_ROOT
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
        chunker = DocumentChunker()

        # 优先级：Bedrock Cohere（USE_BEDROCK_EMBEDDING=true）→ 本地 → OpenAI
        use_bedrock_emb = os.environ.get("USE_BEDROCK_EMBEDDING", "").lower() in ("1", "true", "yes")
        use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        embedding = EmbeddingProvider(
            use_bedrock=use_bedrock_emb,
            use_local=use_local,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            local_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )

        vk = int(os.environ.get("VECTOR_TOP_K", "10"))
        bk = int(os.environ.get("BM25_TOP_K", "10"))
        rk = int(os.environ.get("RERANK_TOP_K", "6"))

        # ---- 主合同 ----
        self._index_one(
            state, chunker, embedding, vk, bk, rk,
            doc_attr="parsed_doc",
            chunks_attr="chunks",
            retriever_attr="retriever",
            contract_id=state.contract_id,
            chunks_path_legacy=Path(state.output_dir) / "chunks.json",
            chroma_persist_legacy=os.environ.get(
                "CHROMA_PERSIST_DIR",
                str(Path(state.output_dir) / "chroma_db"),
            ),
            collection_legacy="contract_docs",
            label="主合同",
        )

        # ---- v2 ----
        if state.parsed_doc_v2 is not None:
            self._index_one(
                state, chunker, embedding, vk, bk, rk,
                doc_attr="parsed_doc_v2",
                chunks_attr="chunks_v2",
                retriever_attr="retriever_v2",
                contract_id=state.contract_id_v2,
                chunks_path_legacy=Path(state.output_dir) / "chunks_v2.json",
                chroma_persist_legacy=str(Path(state.output_dir) / "chroma_db_v2"),
                collection_legacy="contract_docs_v2",
                label="v2 合同",
            )

    # ------------------------------------------------------------------
    def _index_one(
        self, state, chunker, embedding,
        vk, bk, rk,
        doc_attr, chunks_attr, retriever_attr,
        contract_id, chunks_path_legacy: Path,
        chroma_persist_legacy: str, collection_legacy: str,
        label: str,
    ):
        doc = getattr(state, doc_attr, None)
        if doc is None:
            return

        # ---- 路径决策：ContractLibrary 模式 vs 旧模式 ----
        if contract_id:
            lib = ContractLibrary()
            info = lib.get(contract_id)
            if info is None:
                state.errors.append(f"{label}：contract_id={contract_id} 未注册")
                return
            paths = lib.paths(contract_id)
            chunks_path = paths["chunks"]
            chroma_persist = str(DEFAULT_CHROMA_ROOT)
            collection_name = paths["chroma_collection_name"]
            paths["base"].mkdir(parents=True, exist_ok=True)
        else:
            chunks_path = chunks_path_legacy
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            chroma_persist = chroma_persist_legacy
            collection_name = collection_legacy

        # ---- 分块（如果 chunks.json 已存在，直接 load） ----
        chunks = None
        if chunks_path.exists():
            try:
                from chunker import Chunk
                raw = json.loads(chunks_path.read_text(encoding="utf-8"))
                chunks = [
                    Chunk(chunk_id=c["chunk_id"], content=c["content"], metadata=c.get("metadata", {}))
                    for c in raw
                ]
                state.log(self.name, f"{label} 复用 chunks.json", n_chunks=len(chunks))
            except Exception as e:
                state.log(self.name, f"{label} chunks.json 加载失败：{e}", level="warning")
                chunks = None

        if chunks is None:
            chunks = chunker.chunk(doc)
            chunker.save_chunks(chunks, str(chunks_path))
            state.log(self.name, f"{label} 分块完成", n_chunks=len(chunks))

        setattr(state, chunks_attr, chunks)

        # ---- 索引 ----
        retriever = Retriever(
            embedding_provider=embedding,
            persist_dir=chroma_persist,
            vector_top_k=vk, bm25_top_k=bk, rerank_top_k=rk,
            collection_name=collection_name,
        )
        # ContractLibrary 模式下尝试复用已有 collection
        retriever.index(chunks, reuse_existing=bool(contract_id))
        setattr(state, retriever_attr, retriever)

        if contract_id:
            lib = ContractLibrary()
            lib.update(contract_id, chunks=len(chunks), status="indexed")
            lib.touch(contract_id)

        state.log(self.name, f"{label} 索引完成", collection=collection_name)
