"""
文档分块模块

分块策略（混合方式）：
1. 按章节标题层级切分 — 同一节的内容尽量在同一 chunk
2. 表格独立成块 — 不与其他文本混合
3. 段落按语义边界切分 — 超过最大长度的段落递归切分
4. 每个 chunk 携带完整 metadata

Metadata 字段：
- section_path: 章节路径（如"第三章 > 3.2 系统功能范围"）
- block_type: 内容类型（section_title/paragraph/table/list/figure_description/signature）
- page_hint: 页码范围
- table_id: 表格唯一标识（仅表格类型）
- source_text: 原文摘录
- chunk_index: chunk 序号
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pdf_parser import ParsedDocument, TextBlock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """文档分块"""
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    """文档分块器"""

    MAX_CHUNK_CHARS = 1200      # 每个 chunk 最大字符数
    OVERLAP_CHARS = 150          # chunk 间重叠字符数

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """将解析后的文档切分为 chunks"""
        logger.info(f"开始分块，共 {len(parsed_doc.blocks)} 个文本块")

        # 第一步：合并同节内的小段落，拆分过大的段落
        merged_blocks = self._merge_related_blocks(parsed_doc.blocks)

        # 第二步：按大小切分，加入重叠
        chunks = []
        for block in merged_blocks:
            block_chunks = self._split_block(block)
            chunks.extend(block_chunks)

        # 第三步：分配 chunk_id
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
            chunk.chunk_id = f"chunk_{i:04d}_{content_hash}"
            chunk.metadata["chunk_index"] = i

        logger.info(f"分块完成: {len(chunks)} 个 chunks")
        return chunks

    def _merge_related_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """合并在同一节下的连续段落"""
        merged = []
        current = None

        for block in blocks:
            if current is None:
                current = block
            elif (
                current.block_type == "paragraph"
                and block.block_type == "paragraph"
                and current.section_path == block.section_path
                and len(current.content) + len(block.content) < self.MAX_CHUNK_CHARS
            ):
                # 合并连续段落
                current.content = current.content + "\n" + block.content
                current.metadata.update(block.metadata)
            else:
                merged.append(current)
                current = block

        if current:
            merged.append(current)

        return merged

    def _split_block(self, block: TextBlock) -> list[Chunk]:
        """将单个文本块切分为一个或多个 chunk"""
        content = block.content.strip()
        if not content:
            return []

        # 表格：尽量保持完整
        if block.block_type == "table":
            return self._split_table(block)

        # 短内容：直接作为一个 chunk
        if len(content) <= self.MAX_CHUNK_CHARS:
            return [Chunk(
                chunk_id="",  # 后续分配
                content=content,
                metadata={
                    "section_path": block.section_path,
                    "block_type": block.block_type,
                    "page_hint": block.page,
                    "source_text": content,
                    **(block.metadata),
                },
            )]

        # 长内容：按段落/句子边界递归切分
        return self._split_long_content(block)

    def _split_table(self, block: TextBlock) -> list[Chunk]:
        """处理表格分块 — 尽量完整，超长时按行切分"""
        lines = block.content.split('\n')
        content = block.content

        if len(content) <= self.MAX_CHUNK_CHARS:
            return [Chunk(
                chunk_id="",
                content=content,
                metadata={
                    "section_path": block.section_path,
                    "block_type": "table",
                    "page_hint": block.page,
                    "table_id": block.metadata.get("table_id", ""),
                    "source_text": content,
                    "table_rows": len(lines),
                },
            )]

        # 超长表格：按行切分，保留表头
        header = lines[0] if lines else ""
        chunks = []
        current_lines = [header]
        current_len = len(header)

        for line in lines[1:]:
            if current_len + len(line) > self.MAX_CHUNK_CHARS:
                chunks.append(Chunk(
                    chunk_id="",
                    content='\n'.join(current_lines),
                    metadata={
                        "section_path": block.section_path,
                        "block_type": "table",
                        "page_hint": block.page,
                        "table_id": block.metadata.get("table_id", ""),
                        "source_text": '\n'.join(current_lines),
                        "table_partial": True,
                    },
                ))
                current_lines = [header + " (续)"]
                current_len = len(header)
            current_lines.append(line)
            current_len += len(line)

        if len(current_lines) > 1:
            chunks.append(Chunk(
                chunk_id="",
                content='\n'.join(current_lines),
                metadata={
                    "section_path": block.section_path,
                    "block_type": "table",
                    "page_hint": block.page,
                    "table_id": block.metadata.get("table_id", ""),
                    "source_text": '\n'.join(current_lines),
                    "table_partial": True,
                },
            ))

        return chunks

    def _split_long_content(self, block: TextBlock) -> list[Chunk]:
        """按句子/段落边界递归切分长文本"""
        content = block.content
        max_len = self.MAX_CHUNK_CHARS
        overlap = self.OVERLAP_CHARS

        # 按段落分割
        paragraphs = content.split('\n')
        chunks = []
        current_text = ""
        current_source = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_text) + len(para) + 1 <= max_len:
                current_text = (current_text + "\n" + para).strip()
            else:
                if current_text:
                    chunks.append(self._make_chunk(current_text, block))
                    # 重叠：保留上一段最后部分
                    overlap_text = current_text[-overlap:] if len(current_text) > overlap else current_text
                    current_text = overlap_text + "\n" + para
                else:
                    # 单段超过上限，按句子切分
                    sub_chunks = self._split_by_sentences(para, block, max_len, overlap)
                    chunks.extend(sub_chunks)
                    current_text = ""

        if current_text.strip():
            chunks.append(self._make_chunk(current_text, block))

        return chunks

    def _split_by_sentences(self, text: str, block: TextBlock, max_len: int, overlap: int) -> list[Chunk]:
        """按句子边界切分"""
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= max_len:
                current += sent
            else:
                if current.strip():
                    chunks.append(self._make_chunk(current, block))
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    current = overlap_text + sent
                else:
                    # 单句超过上限，硬切
                    for i in range(0, len(sent), max_len - overlap):
                        piece = sent[i:i + max_len]
                        chunks.append(self._make_chunk(piece, block))
                    current = ""

        if current.strip():
            chunks.append(self._make_chunk(current, block))

        return chunks

    def _make_chunk(self, content: str, block: TextBlock) -> Chunk:
        return Chunk(
            chunk_id="",
            content=content,
            metadata={
                "section_path": block.section_path,
                "block_type": block.block_type,
                "page_hint": block.page,
                "source_text": content,
                **(block.metadata),
            },
        )

    def save_chunks(self, chunks: list[Chunk], output_path: str):
        """保存分块结果"""
        data = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"分块结果已保存至: {output_path}")
