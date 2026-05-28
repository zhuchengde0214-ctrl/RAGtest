"""
主入口

流程：
1. PDF 解析（带 per-page 缓存）
2. 文档分块 + 元数据
3. 建索引（dense + sparse）
4. RAG 问答（Q1 / Q2 / Q3）
5. 合同审查（10 维度定向检索）
6. 输出 outputs/qa_results.json + outputs/review_results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

from pdf_parser import PDFParser, load_parsed_document
from chunker import DocumentChunker
from retriever import EmbeddingProvider, Retriever
from qa_engine import QAEngine
from review_engine import ReviewEngine
from llm_client import get_default_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------- 题目 3 类问题 ----------
Q1 = "本项目包含哪些主要系统模块？请列出模块名称，并说明每个模块的主要用途。"
Q2 = [
    "本项目需要交付哪些主要成果物？",
    "这些成果物分别对应哪些验收材料？",
    "如果验收材料缺失，可能影响哪些付款节点？请说明依据。",
]
Q3 = "请综合判断本项目的付款安排、验收条件、交付计划和附件说明之间是否存在冲突或不一致。请逐条说明依据，并指出哪些问题需要人工复核。"


def parse_args():
    p = argparse.ArgumentParser(description="合同 AI 审查与知识库检索")
    p.add_argument("--pdf", type=str, default=os.environ.get("PDF_PATH", "data/AI知识库-综合测试文档.pdf"))
    p.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "outputs"))
    p.add_argument("--cache-dir", type=str, default=None, help="OCR 单页缓存目录，默认 outputs/.ocr_cache")
    p.add_argument("--skip-ocr", action="store_true", help="跳过 OCR，使用已保存的 parsed_document.json")
    p.add_argument("--parsed-json", type=str, default=None)
    p.add_argument("--max-pages", type=int, default=None, help="只处理前 N 页（调试用）")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--no-qa", action="store_true", help="跳过 QA 阶段")
    p.add_argument("--no-review", action="store_true", help="跳过审查阶段")
    return p.parse_args()


def main():
    args = parse_args()

    use_bedrock = os.environ.get("USE_BEDROCK", "false").lower() in ("1", "true", "yes")
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    model = args.model or get_default_model()

    if not use_bedrock:
        if not api_key or api_key.startswith("sk-ant-xxxx"):
            logger.error("请在 .env 配置真实的 ANTHROPIC_API_KEY，或设 USE_BEDROCK=true 使用 AWS Bedrock")
            sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / ".ocr_cache"

    logger.info("=" * 60)
    logger.info("合同 AI — 启动")
    logger.info(f"  backend    : {'AWS Bedrock' if use_bedrock else 'Anthropic API'}")
    logger.info(f"  PDF        : {args.pdf}")
    logger.info(f"  output     : {output_dir}")
    logger.info(f"  cache      : {cache_dir}")
    logger.info(f"  model      : {model}")
    logger.info("=" * 60)

    # ---------- Step 1: 解析 ----------
    if args.skip_ocr and args.parsed_json:
        logger.info("[Step 1] 加载已保存的 parsed_document.json")
        parsed = load_parsed_document(args.parsed_json)
    elif args.skip_ocr and (output_dir / "parsed_document.json").exists():
        path = output_dir / "parsed_document.json"
        logger.info(f"[Step 1] 加载已保存的解析文档: {path}")
        parsed = load_parsed_document(str(path))
    else:
        logger.info("[Step 1] PDF 解析（OCR + 缓存）")
        pdf_parser = PDFParser(api_key=api_key, model=model, cache_dir=str(cache_dir))
        parsed = pdf_parser.parse(args.pdf, max_pages=args.max_pages)
        pdf_parser.save_parsed_document(parsed, str(output_dir / "parsed_document.json"))

    # 全文 dump（便于评审查阅）
    (output_dir / "full_text.txt").write_text(parsed.raw_text, encoding="utf-8")

    # ---------- Step 2: 分块 ----------
    logger.info("[Step 2] 分块")
    chunker = DocumentChunker()
    chunks = chunker.chunk(parsed)
    chunker.save_chunks(chunks, str(output_dir / "chunks.json"))

    # ---------- Step 3: 索引 ----------
    logger.info("[Step 3] 建索引")
    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    embedding = EmbeddingProvider(
        use_local=use_local,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    retriever = Retriever(
        embedding_provider=embedding,
        persist_dir=os.environ.get("CHROMA_PERSIST_DIR", str(output_dir / "chroma_db")),
        vector_top_k=int(os.environ.get("VECTOR_TOP_K", "10")),
        bm25_top_k=int(os.environ.get("BM25_TOP_K", "10")),
        rerank_top_k=int(os.environ.get("RERANK_TOP_K", "6")),
    )
    retriever.index(chunks)

    # ---------- Step 4: QA ----------
    if not args.no_qa:
        logger.info("[Step 4] RAG 问答")
        qa = QAEngine(api_key=api_key, model=model)
        results = []
        results.append(qa.answer_simple(Q1, retriever))
        results.extend(qa.answer_multi_turn(Q2, retriever))
        results.append(qa.answer_complex(Q3, retriever))

        qa_path = output_dir / "qa_results.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"  → {qa_path}")

    # ---------- Step 5: 审查 ----------
    if not args.no_review:
        logger.info("[Step 5] 合同审查")
        engine = ReviewEngine(api_key=api_key, model=model)
        risks = engine.review(chunks, retriever)
        review_path = output_dir / "review_results.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(risks, f, ensure_ascii=False, indent=2)
        logger.info(f"  → {review_path}")

    logger.info("=" * 60)
    logger.info("全部完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
