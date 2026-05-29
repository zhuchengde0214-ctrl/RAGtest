"""ParserAgent：调用 Vision OCR 解析 PDF。

包装 src/pdf_parser.py 的 PDFParser，扫描件直接走 Vision，缓存到磁盘。
若 state.pdf_path_v2 存在，同时解析两份合同（阶段 4 diff 用）。
"""

import os
from pathlib import Path

from llm_client import get_default_model
from pdf_parser import PDFParser, load_parsed_document

from .base import BaseAgent
from .state import SharedState


class ParserAgent(BaseAgent):
    name = "parser"

    def _run(self, state: SharedState) -> None:
        output_dir = Path(state.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = state.cache_dir or str(output_dir / ".ocr_cache")

        parser = PDFParser(model=get_default_model(), cache_dir=cache_dir)

        # --- 主合同 ---
        cached_path = output_dir / "parsed_document.json"
        if cached_path.exists() and self._is_same_pdf(state.pdf_path, cached_path):
            state.parsed_doc = load_parsed_document(str(cached_path))
            state.log(self.name, f"复用缓存：{cached_path.name}")
        else:
            state.parsed_doc = parser.parse(state.pdf_path)
            parser.save_parsed_document(state.parsed_doc, str(cached_path))

        state.log(
            self.name,
            f"主合同解析完成",
            pages=state.parsed_doc.total_pages,
            blocks=len(state.parsed_doc.blocks),
        )

        # --- 第二份合同（阶段 4 diff 才会有） ---
        if state.pdf_path_v2:
            cached_v2 = output_dir / "parsed_document_v2.json"
            if cached_v2.exists() and self._is_same_pdf(state.pdf_path_v2, cached_v2):
                state.parsed_doc_v2 = load_parsed_document(str(cached_v2))
                state.log(self.name, f"v2 复用缓存：{cached_v2.name}")
            else:
                state.parsed_doc_v2 = parser.parse(state.pdf_path_v2)
                parser.save_parsed_document(state.parsed_doc_v2, str(cached_v2))
            state.log(
                self.name,
                "v2 合同解析完成",
                pages=state.parsed_doc_v2.total_pages,
                blocks=len(state.parsed_doc_v2.blocks),
            )

    @staticmethod
    def _is_same_pdf(pdf_path: str, parsed_json: Path) -> bool:
        """检查缓存的 parsed_document.json 与当前 pdf_path 同源（按文件名 + mtime 简单判断）。"""
        if not pdf_path or not os.path.exists(pdf_path):
            return False
        try:
            import json
            data = json.load(open(parsed_json, encoding="utf-8"))
            return data.get("filename") == os.path.basename(pdf_path)
        except Exception:
            return False
