"""
PDF 解析模块 — 扫描件 OCR

流程：
1. PyMuPDF 渲染每页为 300 DPI PNG
2. （可选）尝试原生文本层：如有则跳过 Vision 调用
3. Claude Vision API 提取结构化文本
4. 结构标记解析为 TextBlock（标题/段落/表格/图示/签署）
5. 跨页表格合并：表头继承 + page_hint 关联
6. 单页缓存到 outputs/.ocr_cache/，重跑零成本

Block 类型：
- section_title  章/节标题
- paragraph      普通段落
- list           列表项
- table          表格（独立成块）
- figure         图示/流程图的文字描述
- signature      签署区
- unreadable     无法识别（needs_review=True）
"""

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from llm_client import make_llm_client, get_default_model

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    block_id: str
    block_type: str
    section_path: str
    page: int
    content: str
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0
    needs_review: bool = False


@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    blocks: list[TextBlock]
    raw_text: str

    def page_text(self, page: int) -> str:
        return "\n".join(b.content for b in self.blocks if b.page == page)


class PDFParser:
    VISION_PROMPT = """你正在解析一份中文企业合同的扫描页。请逐字提取页面上所有正文内容，不要总结、不要翻译。

请用以下结构化标记包裹不同类型的内容（每个标记独占一行）：

[PAGE: <页码>]                  ← 如果页面上能看到页码
[TITLE] 一级标题                 ← 章 / 第X章
[SECTION] 二级或三级节标题       ← 1.1 / 3.2.1 等
[LIST] 列表项                    ← 编号或项目符号开头
[TABLE_START] 表格名（可选）
| 列1 | 列2 | 列3 |              ← 用 | 分隔单元格，每行一行
| --- | --- | --- |              ← 表头分隔行（可选）
| 数据 | 数据 | 数据 |
[TABLE_END]
[FIGURE] 流程图/架构图的纯文字描述：依次列出节点、节点之间的箭头方向、关键说明文字
[SIGNATURE] 签署区出现的所有可读文字（甲方/乙方名称、日期、印章上文字）
其他正文不加标记，直接输出，每段一行。

硬性要求：
1. 不要省略任何金额、日期、百分比、章节编号、附件编号
2. 表格必须完整，单元格为空写"-"
3. 忽略页眉页脚水印（如"内部资料""第X页/共Y页"）
4. 印章覆盖、模糊不清的字段写成 [?]
5. 整页完全无法辨认才输出 [UNREADABLE]
6. 直接输出提取结果，不要任何解释"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        dpi: int = 300,
    ):
        self.model = model or get_default_model()
        self.dpi = dpi
        self.client = make_llm_client(api_key=api_key)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 主解析入口
    # ------------------------------------------------------------------
    def parse(
        self,
        pdf_path: str,
        start_page: int = 0,
        max_pages: Optional[int] = None,
    ) -> ParsedDocument:
        logger.info(f"打开 PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        end_page = min(total_pages, start_page + max_pages) if max_pages else total_pages
        logger.info(f"共 {total_pages} 页，处理 {start_page + 1}-{end_page}")

        all_blocks: list[TextBlock] = []
        all_raw: list[str] = []
        # 章节层级状态：跨页延续
        ctx = {"chapter": "", "section": ""}

        for page_num in range(start_page, end_page):
            logger.info(f"  解析第 {page_num + 1}/{end_page} 页")
            raw_text = self._extract_page_text(doc, page_num, pdf_path)
            page_blocks = self._parse_blocks(raw_text, page_num + 1, ctx)
            all_blocks.extend(page_blocks)
            all_raw.append(raw_text)

        doc.close()

        # 跨页表格合并（表头继承）
        all_blocks = self._merge_cross_page_tables(all_blocks)

        parsed = ParsedDocument(
            filename=os.path.basename(pdf_path),
            total_pages=total_pages,
            blocks=all_blocks,
            raw_text="\n\n".join(all_raw),
        )
        logger.info(f"解析完成: {len(all_blocks)} 个 block")
        return parsed

    # ------------------------------------------------------------------
    # 单页解析（带缓存）
    # ------------------------------------------------------------------
    def _extract_page_text(self, doc, page_num: int, pdf_path: str) -> str:
        cache_key = self._cache_key(pdf_path, page_num)
        if self.cache_dir:
            cached = self.cache_dir / f"{cache_key}.txt"
            if cached.exists():
                logger.info(f"    使用缓存: {cached.name}")
                return cached.read_text(encoding="utf-8")

        page = doc[page_num]
        # 优先尝试原生文本层
        native = page.get_text().strip()
        if native and len(native) > 50:
            text = "[NATIVE]\n" + native
        else:
            # 渲染为图像走 Vision OCR
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            text = self._call_vision_api(img_bytes)

        if self.cache_dir:
            (self.cache_dir / f"{cache_key}.txt").write_text(text, encoding="utf-8")
        return text

    def _cache_key(self, pdf_path: str, page_num: int) -> str:
        h = hashlib.md5(f"{os.path.basename(pdf_path)}::{page_num}::{self.dpi}".encode()).hexdigest()[:10]
        return f"p{page_num + 1:03d}_{h}"

    def _call_vision_api(self, img_bytes: bytes) -> str:
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
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
            return msg.content[0].text
        except Exception as e:
            logger.error(f"Vision API 调用失败: {e}")
            return "[UNREADABLE]"

    # ------------------------------------------------------------------
    # 结构化标记解析
    # ------------------------------------------------------------------
    def _parse_blocks(self, raw_text: str, page_num: int, ctx: dict) -> list[TextBlock]:
        if "[UNREADABLE]" in raw_text:
            logger.warning(f"  第 {page_num} 页无法识别")
            return [TextBlock(
                block_id=f"p{page_num}_unreadable",
                block_type="unreadable",
                section_path=self._build_section_path(ctx),
                page=page_num,
                content="[此页无法识别，需人工复核]",
                confidence=0.0,
                needs_review=True,
            )]

        # 原生文本层路径：整页作一个段落，由 chunker 后续切分
        if raw_text.startswith("[NATIVE]"):
            content = raw_text[len("[NATIVE]"):].strip()
            return [TextBlock(
                block_id=f"p{page_num}_native",
                block_type="paragraph",
                section_path=self._build_section_path(ctx),
                page=page_num,
                content=content,
                metadata={"source": "native_text_layer"},
            )]

        blocks: list[TextBlock] = []
        block_idx = 0
        in_table = False
        table_buf: list[str] = []
        table_caption = ""

        for line in raw_text.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue

            # 处理 [PAGE: N] 标记（仅作信息，不入 block）
            if re.match(r"\[PAGE:\s*\d+\]", line):
                continue

            if in_table:
                if line.startswith("[TABLE_END]"):
                    blocks.append(TextBlock(
                        block_id=f"p{page_num}_b{block_idx}",
                        block_type="table",
                        section_path=self._build_section_path(ctx),
                        page=page_num,
                        content="\n".join(table_buf),
                        metadata={
                            "table_id": f"tbl_p{page_num}_b{block_idx}",
                            "table_caption": table_caption,
                            "table_rows": len(table_buf),
                        },
                    ))
                    block_idx += 1
                    in_table = False
                    table_buf = []
                    table_caption = ""
                else:
                    table_buf.append(line)
                continue

            if line.startswith("[TITLE]"):
                title = line[len("[TITLE]"):].strip()
                ctx["chapter"] = title
                ctx["section"] = ""
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="section_title",
                    section_path=title,
                    page=page_num,
                    content=title,
                    metadata={"level": 1},
                ))
                block_idx += 1
            elif line.startswith("[SECTION]"):
                sec = line[len("[SECTION]"):].strip()
                ctx["section"] = sec
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="section_title",
                    section_path=self._build_section_path(ctx),
                    page=page_num,
                    content=sec,
                    metadata={"level": 2},
                ))
                block_idx += 1
            elif line.startswith("[TABLE_START]"):
                table_caption = line[len("[TABLE_START]"):].strip()
                in_table = True
                table_buf = []
            elif line.startswith("[FIGURE]"):
                content = line[len("[FIGURE]"):].strip()
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="figure",
                    section_path=self._build_section_path(ctx),
                    page=page_num,
                    content=content,
                ))
                block_idx += 1
            elif line.startswith("[SIGNATURE]"):
                content = line[len("[SIGNATURE]"):].strip()
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="signature",
                    section_path=self._build_section_path(ctx),
                    page=page_num,
                    content=content,
                    needs_review="[?]" in content,
                ))
                block_idx += 1
            elif line.startswith("[LIST]"):
                content = line[len("[LIST]"):].strip()
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="list",
                    section_path=self._build_section_path(ctx),
                    page=page_num,
                    content=content,
                ))
                block_idx += 1
            else:
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{block_idx}",
                    block_type="paragraph",
                    section_path=self._build_section_path(ctx),
                    page=page_num,
                    content=line.strip(),
                    needs_review="[?]" in line,
                ))
                block_idx += 1

        # 兜底：如果 [TABLE_START] 没有对应的 [TABLE_END]
        if in_table and table_buf:
            blocks.append(TextBlock(
                block_id=f"p{page_num}_b{block_idx}",
                block_type="table",
                section_path=self._build_section_path(ctx),
                page=page_num,
                content="\n".join(table_buf),
                metadata={
                    "table_id": f"tbl_p{page_num}_b{block_idx}",
                    "table_caption": table_caption,
                    "table_partial_unclosed": True,
                },
                needs_review=True,
            ))

        return blocks

    @staticmethod
    def _build_section_path(ctx: dict) -> str:
        parts = [p for p in (ctx.get("chapter"), ctx.get("section")) if p]
        return " > ".join(parts)

    # ------------------------------------------------------------------
    # 跨页表格合并（同一节、相邻页、表头一致 → 合并）
    # ------------------------------------------------------------------
    def _merge_cross_page_tables(self, blocks: list[TextBlock]) -> list[TextBlock]:
        if not blocks:
            return blocks
        merged: list[TextBlock] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            if cur.block_type != "table":
                merged.append(cur)
                i += 1
                continue
            # 向后找连续的同节表格
            j = i + 1
            chain = [cur]
            while j < len(blocks):
                nxt = blocks[j]
                if (
                    nxt.block_type == "table"
                    and nxt.section_path == cur.section_path
                    and nxt.page in (chain[-1].page, chain[-1].page + 1)
                ):
                    chain.append(nxt)
                    j += 1
                elif nxt.block_type == "section_title":
                    break
                else:
                    # 非表格、非节标题的内容打断了表格链
                    break
            if len(chain) > 1 and self._tables_share_header(chain):
                first = chain[0]
                first_lines = first.content.splitlines()
                header = first_lines[0] if first_lines else ""
                merged_content = first.content
                for t in chain[1:]:
                    extra = t.content.splitlines()
                    # 续表去掉重复表头
                    if extra and extra[0].strip() == header.strip():
                        extra = extra[1:]
                    if extra:
                        merged_content += "\n" + "\n".join(extra)
                merged_block = TextBlock(
                    block_id=first.block_id + "_merged",
                    block_type="table",
                    section_path=first.section_path,
                    page=first.page,
                    content=merged_content,
                    metadata={
                        "table_id": first.metadata.get("table_id", ""),
                        "table_caption": first.metadata.get("table_caption", ""),
                        "table_rows": len(merged_content.splitlines()),
                        "table_pages": [t.page for t in chain],
                        "table_merged_from": [t.block_id for t in chain],
                    },
                )
                merged.append(merged_block)
                i = j
            else:
                merged.append(cur)
                i += 1
        return merged

    @staticmethod
    def _tables_share_header(chain: list[TextBlock]) -> bool:
        if not chain:
            return False
        first_header = chain[0].content.splitlines()[0].strip() if chain[0].content else ""
        if not first_header.startswith("|"):
            return False
        for t in chain[1:]:
            lines = t.content.splitlines()
            if not lines:
                return False
            # 后续表格首行要么是相同表头、要么直接是数据行（无表头）
            if lines[0].strip() == first_header:
                continue
            if lines[0].strip().startswith("|"):
                continue
            return False
        return True

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save_parsed_document(self, parsed: ParsedDocument, output_path: str):
        data = {
            "filename": parsed.filename,
            "total_pages": parsed.total_pages,
            "blocks": [asdict(b) for b in parsed.blocks],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"解析结果已保存: {output_path}")


def load_parsed_document(json_path: str) -> ParsedDocument:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = [
        TextBlock(
            block_id=b["block_id"],
            block_type=b["block_type"],
            section_path=b.get("section_path", ""),
            page=b["page"],
            content=b["content"],
            metadata=b.get("metadata", {}) or {},
            confidence=b.get("confidence", 1.0),
            needs_review=b.get("needs_review", False),
        )
        for b in data["blocks"]
    ]
    raw = "\n\n".join(b.content for b in blocks)
    return ParsedDocument(
        filename=data["filename"],
        total_pages=data["total_pages"],
        blocks=blocks,
        raw_text=raw,
    )
