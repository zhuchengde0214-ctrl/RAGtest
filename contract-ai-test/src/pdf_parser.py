"""
PDF 文档解析模块

处理扫描件 PDF：
1. 使用 PyMuPDF (fitz) 将每页渲染为图像
2. 使用 Claude Vision API 从图像中提取结构化文本
3. 返回带页码、区块类型标记的结构化文档

失败边界：
- 若页面完全无法识别（模糊/遮挡），标记为低置信度并进入人工复核
- 若表格数据提取不完整，记录警告
"""

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """文档中的文本块"""
    block_id: str
    block_type: str          # section_title | paragraph | table | list | figure_description | signature
    section_path: str        # 如 "第三章 > 3.2 系统功能范围"
    page: int
    content: str
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    """解析后的完整文档"""
    filename: str
    total_pages: int
    blocks: list[TextBlock]
    raw_text: str


class PDFParser:
    """PDF 解析器 — 扫描件 OCR 流程"""

    VISION_PROMPT = """你是一个专业的文档解析助手。请仔细阅读这张扫描文档页面，提取其中所有文字内容。

要求：
1. 按阅读顺序输出所有文字，保持原文内容不变
2. 区分以下内容类型，用标记包裹：
   - [TITLE] 章标题
   - [SECTION] 节标题
   - [TABLE_START] ... [TABLE_END] 表格内容（用 | 分隔单元格，用 --- 分隔表头）
   - [LIST] 列表项
   - [FIGURE] 图示/流程图的文字描述（描述图中包含哪些节点和箭头关系）
   - [SIGNATURE] 签署区域
   - 其余为普通段落
3. 表格必须保持行列结构，每行用换行分隔
4. 如果页面有页码，请在开头标注 [PAGE: N]
5. 如果页面完全无法辨认，输出 [UNREADABLE]

请直接输出提取的内容，不要添加额外解释。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY 未设置")
        self.client = Anthropic(api_key=self.api_key)

    def parse(self, pdf_path: str, start_page: int = 0, max_pages: Optional[int] = None) -> ParsedDocument:
        """解析 PDF 文件，返回结构化文档"""
        logger.info(f"开始解析 PDF: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        end_page = min(total_pages, start_page + max_pages) if max_pages else total_pages

        logger.info(f"PDF 共 {total_pages} 页，处理范围: {start_page + 1}-{end_page}")

        all_blocks = []
        all_raw_text = []

        for page_num in range(start_page, end_page):
            logger.info(f"处理第 {page_num + 1}/{end_page} 页...")
            page_blocks, page_text = self._parse_page(doc, page_num)
            all_blocks.extend(page_blocks)
            all_raw_text.append(page_text)

        doc.close()

        parsed = ParsedDocument(
            filename=os.path.basename(pdf_path),
            total_pages=total_pages,
            blocks=all_blocks,
            raw_text="\n\n".join(all_raw_text),
        )

        logger.info(f"解析完成: {len(all_blocks)} 个文本块")
        return parsed

    def _parse_page(self, doc: fitz.Document, page_num: int) -> tuple[list[TextBlock], str]:
        """解析单页"""
        page = doc[page_num]

        # 渲染页面为图像 (300 DPI)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # 调用 Claude Vision 提取文本
        extracted_text = self._call_vision_api(img_bytes)

        if "[UNREADABLE]" in extracted_text:
            logger.warning(f"第 {page_num + 1} 页无法识别")
            block = TextBlock(
                block_id=f"page_{page_num + 1}_unreadable",
                block_type="unreadable",
                section_path="",
                page=page_num + 1,
                content="[此页无法识别，需人工复核]",
                confidence=0.0,
            )
            return [block], ""

        # 解析结构化标记，拆分为 blocks
        blocks = self._parse_blocks(extracted_text, page_num + 1)
        return blocks, extracted_text

    def _call_vision_api(self, img_bytes: bytes) -> str:
        """调用 Claude Vision API 提取文本"""
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": self.VISION_PROMPT},
                    ],
                }
            ],
        )

        return message.content[0].text

    def _parse_blocks(self, raw_text: str, page_num: int) -> list[TextBlock]:
        """将 vision 返回的结构化文本解析为 TextBlock 列表"""
        blocks = []
        block_id_prefix = f"p{page_num}"

        # 提取页码
        page_match = re.match(r'\[PAGE:\s*(\d+)\]', raw_text)
        actual_page = int(page_match.group(1)) if page_match else page_num

        # 当前 section_path 跟踪
        current_section = ""

        lines = raw_text.split('\n')
        i = 0
        in_table = False
        table_lines = []
        block_index = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # 章节标题
            if line.startswith('[TITLE]'):
                content = line.replace('[TITLE]', '').strip()
                current_section = content
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="section_title",
                    section_path=current_section,
                    page=actual_page,
                    content=content,
                ))
                block_index += 1

            elif line.startswith('[SECTION]'):
                content = line.replace('[SECTION]', '').strip()
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="section_title",
                    section_path=f"{current_section} > {content}" if current_section else content,
                    page=actual_page,
                    content=content,
                ))
                block_index += 1

            elif line.startswith('[TABLE_START]'):
                in_table = True
                table_lines = []

            elif line.startswith('[TABLE_END]'):
                in_table = False
                table_content = '\n'.join(table_lines)
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="table",
                    section_path=current_section,
                    page=actual_page,
                    content=table_content,
                    metadata={"table_id": f"table_p{actual_page}_b{block_index}"},
                ))
                block_index += 1

            elif in_table:
                table_lines.append(line)

            elif line.startswith('[FIGURE]'):
                content = line.replace('[FIGURE]', '').strip()
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="figure_description",
                    section_path=current_section,
                    page=actual_page,
                    content=content,
                ))
                block_index += 1

            elif line.startswith('[SIGNATURE]'):
                content = line.replace('[SIGNATURE]', '').strip()
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="signature",
                    section_path=current_section,
                    page=actual_page,
                    content=content,
                ))
                block_index += 1

            elif line.startswith('[LIST]'):
                content = line.replace('[LIST]', '').strip()
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="list",
                    section_path=current_section,
                    page=actual_page,
                    content=content,
                ))
                block_index += 1

            else:
                # 普通段落
                blocks.append(TextBlock(
                    block_id=f"{block_id_prefix}_b{block_index}",
                    block_type="paragraph",
                    section_path=current_section,
                    page=actual_page,
                    content=line,
                ))
                block_index += 1

            i += 1

        return blocks

    def save_parsed_document(self, parsed: ParsedDocument, output_path: str):
        """保存解析结果为 JSON"""
        data = {
            "filename": parsed.filename,
            "total_pages": parsed.total_pages,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "block_type": b.block_type,
                    "section_path": b.section_path,
                    "page": b.page,
                    "content": b.content,
                    "metadata": b.metadata,
                    "confidence": b.confidence,
                }
                for b in parsed.blocks
            ],
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"解析结果已保存至: {output_path}")


def load_parsed_document(json_path: str) -> ParsedDocument:
    """从 JSON 加载已解析的文档"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    blocks = [
        TextBlock(
            block_id=b["block_id"],
            block_type=b["block_type"],
            section_path=b.get("section_path", ""),
            page=b["page"],
            content=b["content"],
            metadata=b.get("metadata", {}),
            confidence=b.get("confidence", 1.0),
        )
        for b in data["blocks"]
    ]

    return ParsedDocument(
        filename=data["filename"],
        total_pages=data["total_pages"],
        blocks=blocks,
        raw_text="\n\n".join(b.content for b in blocks),
    )
