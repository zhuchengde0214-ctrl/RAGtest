"""ParserAgent：调用 Vision OCR 解析 PDF。

支持两种模式：
1. ContractLibrary 模式（推荐）
   state.contract_id 给定 → 用 lib.paths(contract_id) 决定文件位置
   产物落到 outputs/contracts/<id>/parsed_document.json
   OCR 缓存落到 outputs/contracts/<id>/ocr_cache/
2. 旧的 output_dir 模式（向后兼容 CLI / eval）
   产物落到 state.output_dir/parsed_document.json

若 state.pdf_path_v2 / contract_id_v2 存在，同时解析 v2（diff 用）。
"""

import os
from pathlib import Path

from contract_library import ContractLibrary
from llm_client import get_default_model
from pdf_parser import PDFParser, load_parsed_document

from .base import BaseAgent
from .state import SharedState


class ParserAgent(BaseAgent):
    name = "parser"

    def _run(self, state: SharedState) -> None:
        parser = PDFParser(model=get_default_model())

        # ---- 主合同 ----
        self._parse_one(
            state, parser,
            pdf_path=state.pdf_path,
            contract_id=state.contract_id,
            doc_attr="parsed_doc",
            label="主合同",
        )

        # ---- v2（如果有）----
        if state.pdf_path_v2 or state.contract_id_v2:
            self._parse_one(
                state, parser,
                pdf_path=state.pdf_path_v2 or "",
                contract_id=state.contract_id_v2,
                doc_attr="parsed_doc_v2",
                label="v2 合同",
            )
        else:
            # 向后兼容：旧逻辑，看 outputs/parsed_document_v2.json 是否存在
            cached_v2 = Path(state.output_dir) / "parsed_document_v2.json"
            if cached_v2.exists():
                state.parsed_doc_v2 = load_parsed_document(str(cached_v2))
                state.log(self.name, f"v2 复用旧版缓存：{cached_v2.name}")

        if state.parsed_doc is not None:
            state.log(
                self.name,
                "主合同就绪",
                pages=state.parsed_doc.total_pages,
                blocks=len(state.parsed_doc.blocks),
            )
        if state.parsed_doc_v2 is not None:
            state.log(
                self.name,
                "v2 合同就绪",
                pages=state.parsed_doc_v2.total_pages,
                blocks=len(state.parsed_doc_v2.blocks),
            )

    # ------------------------------------------------------------------
    def _parse_one(
        self, state: SharedState, parser: PDFParser,
        pdf_path: str, contract_id: "str | None",
        doc_attr: str, label: str,
    ):
        """解析一份 PDF，写到 state.<doc_attr>。优先 ContractLibrary 模式。"""
        if contract_id:
            lib = ContractLibrary()
            info = lib.get(contract_id)
            if info is None:
                state.log(self.name, f"{label}：未找到 contract_id={contract_id}", level="error")
                state.errors.append(f"{label}：contract_id 未注册")
                return

            paths = lib.paths(contract_id)
            parsed_path = paths["parsed_document"]
            ocr_cache = paths["ocr_cache"]
            ocr_cache.mkdir(parents=True, exist_ok=True)

            if parsed_path.exists():
                doc = load_parsed_document(str(parsed_path))
                setattr(state, doc_attr, doc)
                lib.touch(contract_id)
                state.log(self.name, f"{label} 复用长期记忆：{paths['base'].name}/parsed_document.json")
                return

            # 需要重新 OCR
            actual_pdf = info.pdf_path
            parser.cache_dir = ocr_cache  # 覆盖 cache_dir
            doc = parser.parse(actual_pdf)
            parser.save_parsed_document(doc, str(parsed_path))
            lib.update(
                contract_id,
                pages=doc.total_pages,
                status="parsed",
            )
            setattr(state, doc_attr, doc)
            return

        # 旧模式：直接用 pdf_path + state.output_dir
        if not pdf_path:
            state.log(self.name, f"{label}：无 pdf_path 也无 contract_id，跳过", level="warning")
            return

        output_dir = Path(state.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = state.cache_dir or str(output_dir / ".ocr_cache")
        parser.cache_dir = Path(cache_dir)
        parser.cache_dir.mkdir(parents=True, exist_ok=True)

        cached_path = output_dir / ("parsed_document.json" if doc_attr == "parsed_doc" else "parsed_document_v2.json")
        if cached_path.exists() and self._is_same_pdf(pdf_path, cached_path):
            doc = load_parsed_document(str(cached_path))
            state.log(self.name, f"{label} 复用缓存：{cached_path.name}")
        else:
            doc = parser.parse(pdf_path)
            parser.save_parsed_document(doc, str(cached_path))
        setattr(state, doc_attr, doc)

    @staticmethod
    def _is_same_pdf(pdf_path: str, parsed_json: Path) -> bool:
        if not pdf_path or not os.path.exists(pdf_path):
            return False
        try:
            import json
            data = json.load(open(parsed_json, encoding="utf-8"))
            return data.get("filename") == os.path.basename(pdf_path)
        except Exception:
            return False
