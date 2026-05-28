"""
文档分块模块

策略（混合）：
1. 表格、图示、签署区独立成块（保留完整结构）
2. 同 section_path 下的连续 paragraph/list 合并到 ≤ MAX_CHUNK_CHARS
3. 超长内容按句号边界递归切分，相邻 chunk 之间保留 OVERLAP_CHARS
4. 不修改原 TextBlock，每个 chunk 重新组装

每个 chunk 的 metadata：
- chunk_index: 全局序号
- block_ids: 来源 block_id 列表（用于回溯）
- section_path
- block_type: paragraph / table / figure / signature / mixed
- page_hint: 主页码
- pages: 跨页时的页码列表
- table_id / table_caption: 仅 table
- needs_review: 来源 block 任一为 True 则继承
- char_len
- source_text: chunk 完整原文
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pdf_parser import ParsedDocument, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    MAX_CHUNK_CHARS = 1000
    MIN_MERGE_CHARS = 600    # 小于此长度的同节段落会被尝试合并
    OVERLAP_CHARS = 120

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        logger.info(f"开始分块: {len(parsed_doc.blocks)} 个 block")

        chunks: list[Chunk] = []
        # 先按 block 顺序遍历，把可合并的连续段落聚成 group
        groups = self._group_blocks(parsed_doc.blocks)

        for group in groups:
            chunks.extend(self._chunks_from_group(group))

        # 分配 chunk_id
        for i, c in enumerate(chunks):
            h = hashlib.md5(c.content.encode("utf-8")).hexdigest()[:8]
            c.chunk_id = f"chunk_{i:04d}_{h}"
            c.metadata["chunk_index"] = i

        logger.info(f"分块完成: {len(chunks)} 个 chunk")
        return chunks

    # ------------------------------------------------------------------
    # 分组
    # ------------------------------------------------------------------
    def _group_blocks(self, blocks: list[TextBlock]) -> list[list[TextBlock]]:
        """将连续的 paragraph/list/section_title 在同一 section_path 下分到一组。
        独立类型（table/figure/signature/unreadable）单独一组。"""
        STANDALONE = {"table", "figure", "signature", "unreadable"}
        groups: list[list[TextBlock]] = []
        cur: list[TextBlock] = []

        def flush():
            nonlocal cur
            if cur:
                groups.append(cur)
                cur = []

        for b in blocks:
            if b.block_type in STANDALONE:
                flush()
                groups.append([b])
                continue
            if b.block_type == "section_title":
                # 节标题作为下一组的起始
                flush()
                cur = [b]
                continue
            if cur and cur[0].section_path == b.section_path:
                cur.append(b)
            else:
                flush()
                cur = [b]

        flush()
        return groups

    # ------------------------------------------------------------------
    # 单组转 chunk
    # ------------------------------------------------------------------
    def _chunks_from_group(self, group: list[TextBlock]) -> list[Chunk]:
        if not group:
            return []
        first = group[0]

        # 独立块
        if first.block_type == "table":
            return self._chunk_table(first)
        if first.block_type in ("figure", "signature", "unreadable"):
            return [self._make_chunk(
                content=first.content,
                section_path=first.section_path,
                block_type=first.block_type,
                page_hint=first.page,
                pages=[first.page],
                block_ids=[first.block_id],
                needs_review=first.needs_review,
                extra_meta=first.metadata,
            )]

        # 段落/列表/小节标题混合组：拼接后分块
        return self._chunk_text_group(group)

    def _chunk_table(self, block: TextBlock) -> list[Chunk]:
        content = block.content
        meta = block.metadata or {}
        common = {
            "section_path": block.section_path,
            "block_type": "table",
            "page_hint": block.page,
            "pages": meta.get("table_pages", [block.page]),
            "block_ids": [block.block_id],
            "needs_review": block.needs_review,
            "extra_meta": {
                "table_id": meta.get("table_id", ""),
                "table_caption": meta.get("table_caption", ""),
                "table_rows": meta.get("table_rows", len(content.splitlines())),
            },
        }

        if len(content) <= self.MAX_CHUNK_CHARS:
            return [self._make_chunk(content=content, **common)]

        # 大表格：保留表头逐段切
        lines = content.splitlines()
        header = lines[0] if lines else ""
        sep_line = lines[1] if len(lines) > 1 and re.match(r"^\s*\|[\s\-|]+\|\s*$", lines[1]) else None
        body_start = 2 if sep_line else 1
        prefix = "\n".join(lines[:body_start]) + "\n" if body_start else ""

        out: list[Chunk] = []
        cur_lines: list[str] = []
        cur_len = len(prefix)
        for ln in lines[body_start:]:
            if cur_len + len(ln) + 1 > self.MAX_CHUNK_CHARS and cur_lines:
                out.append(self._make_chunk(
                    content=prefix + "\n".join(cur_lines),
                    **{**common, "extra_meta": {**common["extra_meta"], "table_partial": True}},
                ))
                cur_lines = []
                cur_len = len(prefix)
            cur_lines.append(ln)
            cur_len += len(ln) + 1
        if cur_lines:
            out.append(self._make_chunk(
                content=prefix + "\n".join(cur_lines),
                **{**common, "extra_meta": {**common["extra_meta"], "table_partial": len(out) > 0}},
            ))
        return out

    def _chunk_text_group(self, group: list[TextBlock]) -> list[Chunk]:
        # 把组内每个 block 的内容连同其 block_id 一起记录，便于回溯
        pieces: list[tuple[str, TextBlock]] = []
        for b in group:
            text = b.content.strip()
            if not text:
                continue
            if b.block_type == "section_title":
                # 标题加粗放在最前
                pieces.append((f"## {text}", b))
            elif b.block_type == "list":
                pieces.append((f"- {text}", b))
            else:
                pieces.append((text, b))

        if not pieces:
            return []

        section_path = group[0].section_path
        out: list[Chunk] = []
        buf: list[str] = []
        buf_blocks: list[TextBlock] = []
        buf_len = 0

        def emit():
            nonlocal buf, buf_blocks, buf_len
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                buf, buf_blocks, buf_len = [], [], 0
                return
            pages = sorted({b.page for b in buf_blocks})
            block_types = {b.block_type for b in buf_blocks}
            btype = "mixed" if len(block_types) > 1 else next(iter(block_types))
            out.append(self._make_chunk(
                content=content,
                section_path=section_path,
                block_type=btype,
                page_hint=buf_blocks[0].page,
                pages=pages,
                block_ids=[b.block_id for b in buf_blocks],
                needs_review=any(b.needs_review for b in buf_blocks),
            ))
            # overlap：保留尾部最后一段进入下一 chunk
            if self.OVERLAP_CHARS > 0 and len(content) > self.OVERLAP_CHARS:
                tail = content[-self.OVERLAP_CHARS:]
                buf = [tail]
                buf_blocks = [buf_blocks[-1]]
                buf_len = len(tail)
            else:
                buf, buf_blocks, buf_len = [], [], 0

        for text, b in pieces:
            if buf_len + len(text) + 1 > self.MAX_CHUNK_CHARS and buf_len >= self.MIN_MERGE_CHARS:
                emit()
            # 单个 piece 自身超长：先 emit 再按句切
            if len(text) > self.MAX_CHUNK_CHARS:
                if buf:
                    emit()
                for sub in self._split_long_text(text):
                    out.append(self._make_chunk(
                        content=sub,
                        section_path=section_path,
                        block_type=b.block_type,
                        page_hint=b.page,
                        pages=[b.page],
                        block_ids=[b.block_id],
                        needs_review=b.needs_review,
                    ))
                continue
            buf.append(text)
            buf_blocks.append(b)
            buf_len += len(text) + 1

        emit()
        return out

    def _split_long_text(self, text: str) -> list[str]:
        # 按句号/换行切
        sentences = re.split(r"(?<=[。！？!?\.\n])\s*", text)
        out: list[str] = []
        cur = ""
        for s in sentences:
            if not s:
                continue
            if len(cur) + len(s) <= self.MAX_CHUNK_CHARS:
                cur += s
            else:
                if cur.strip():
                    out.append(cur.strip())
                # 单句超长则硬切
                while len(s) > self.MAX_CHUNK_CHARS:
                    out.append(s[: self.MAX_CHUNK_CHARS])
                    s = s[self.MAX_CHUNK_CHARS - self.OVERLAP_CHARS:]
                cur = s
        if cur.strip():
            out.append(cur.strip())
        return out

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _make_chunk(
        self,
        content: str,
        section_path: str,
        block_type: str,
        page_hint: int,
        pages: list[int],
        block_ids: list[str],
        needs_review: bool = False,
        extra_meta: Optional[dict] = None,
    ) -> Chunk:
        meta = {
            "section_path": section_path,
            "block_type": block_type,
            "page_hint": page_hint,
            "pages": list(pages),
            "block_ids": list(block_ids),
            "needs_review": bool(needs_review),
            "char_len": len(content),
            "source_text": content,
        }
        if extra_meta:
            for k, v in extra_meta.items():
                if v is not None and v != "":
                    meta[k] = v
        return Chunk(chunk_id="", content=content, metadata=meta)

    def save_chunks(self, chunks: list[Chunk], output_path: str):
        data = [
            {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
            for c in chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"分块结果已保存: {output_path}")
