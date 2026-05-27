"""
主运行脚本 — 合同 AI 审查与知识库检索

流程：
1. PDF 解析（OCR/视觉模型提取文本）
2. 文档分块 + 元数据标注
3. 建立索引（向量 + BM25）
4. RAG 问答（Q1 简单 / Q2 多轮 / Q3 复杂推理）
5. 合同审查（风险识别）
6. 输出结果文件
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 确保 src 目录在 sys.path 中
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dotenv import load_dotenv

from pdf_parser import PDFParser, load_parsed_document
from chunker import DocumentChunker
from retriever import EmbeddingProvider, Retriever
from qa_engine import QAEngine
from review_engine import ReviewEngine

# 加载 .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 问题定义
# ============================================================

Q1_QUESTION = "本项目包含哪些主要系统模块？请列出模块名称，并说明每个模块的主要用途。"

Q2_QUESTIONS = [
    "本项目需要交付哪些主要成果物？",
    "这些成果物分别对应哪些验收材料？",
    "如果验收材料缺失，可能影响哪些付款节点？请说明依据。",
]

Q3_QUESTION = "请综合判断本项目的付款安排、验收条件、交付计划和附件说明之间是否存在冲突或不一致。请逐条说明依据，并指出哪些问题需要人工复核。"


def main():
    parser = argparse.ArgumentParser(description="合同 AI 审查与知识库检索")
    parser.add_argument("--pdf", type=str, required=True, help="PDF 文件路径")
    parser.add_argument("--output-dir", type=str, default="../outputs", help="输出目录")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR，使用已保存的解析结果")
    parser.add_argument("--parsed-json", type=str, default=None, help="已解析文档 JSON 路径")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API Key")
    parser.add_argument("--model", type=str, default=None, help="Claude 模型名称")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    model = args.model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    if not api_key:
        logger.error("请设置 ANTHROPIC_API_KEY 环境变量或通过 --api-key 参数提供")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("合同 AI 审查与知识库检索 — 开始运行")
    logger.info(f"模型: {model}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # ============================================================
    # Step 1: PDF 解析
    # ============================================================
    if args.skip_ocr and args.parsed_json:
        logger.info("Step 1: 加载已保存的解析文档")
        parsed_doc = load_parsed_document(args.parsed_json)
    else:
        logger.info("Step 1: PDF OCR 解析")
        pdf_parser = PDFParser(api_key=api_key, model=model)
        parsed_doc = pdf_parser.parse(args.pdf)

        # 保存解析结果
        parsed_json_path = output_dir / "parsed_document.json"
        pdf_parser.save_parsed_document(parsed_doc, str(parsed_json_path))

    # 保存完整文本
    full_text_path = output_dir / "full_text.txt"
    with open(full_text_path, 'w', encoding='utf-8') as f:
        f.write(parsed_doc.raw_text)
    logger.info(f"完整文本已保存至: {full_text_path}")

    # ============================================================
    # Step 2: 文档分块
    # ============================================================
    logger.info("Step 2: 文档分块")
    chunker = DocumentChunker()
    chunks = chunker.chunk(parsed_doc)

    # 保存分块结果
    chunks_json_path = output_dir / "chunks.json"
    chunker.save_chunks(chunks, str(chunks_json_path))

    # ============================================================
    # Step 3: 建立索引
    # ============================================================
    logger.info("Step 3: 建立检索索引")

    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    embedding_provider = EmbeddingProvider(
        use_local=use_local,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        persist_dir=os.environ.get("CHROMA_PERSIST_DIR", str(output_dir / "chroma_db")),
        vector_top_k=int(os.environ.get("VECTOR_TOP_K", "8")),
        bm25_top_k=int(os.environ.get("BM25_TOP_K", "8")),
        rerank_top_k=int(os.environ.get("RERANK_TOP_K", "5")),
        hybrid_weight_vector=float(os.environ.get("HYBRID_WEIGHT_VECTOR", "0.5")),
        hybrid_weight_bm25=float(os.environ.get("HYBRID_WEIGHT_BM25", "0.5")),
    )
    retriever.index(chunks)

    # ============================================================
    # Step 4: RAG 问答
    # ============================================================
    logger.info("Step 4: RAG 问答")
    qa_engine = QAEngine(api_key=api_key, model=model)

    all_qa_results = []

    # Q1: 简单事实问题
    logger.info("--- Q1: 简单事实问题 ---")
    q1_result = qa_engine.answer_simple(Q1_QUESTION, retriever)
    all_qa_results.append(q1_result)

    # Q2: 多轮问答
    logger.info("--- Q2: 多轮问答 ---")
    q2_results = qa_engine.answer_multi_turn(Q2_QUESTIONS, retriever)
    all_qa_results.extend(q2_results)

    # Q3: 全文复杂推理
    logger.info("--- Q3: 全文复杂推理 ---")
    q3_result = qa_engine.answer_complex(Q3_QUESTION, retriever)
    all_qa_results.append(q3_result)

    # 保存 QA 结果
    qa_results_path = output_dir / "qa_results.json"
    with open(qa_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_results, f, ensure_ascii=False, indent=2)
    logger.info(f"QA 结果已保存至: {qa_results_path}")

    # ============================================================
    # Step 5: 合同审查
    # ============================================================
    logger.info("Step 5: 合同审查")
    review_engine = ReviewEngine(api_key=api_key, model=model)
    risks = review_engine.review(parsed_doc.raw_text, retriever=retriever)

    # 保存审查结果
    review_results_path = output_dir / "review_results.json"
    with open(review_results_path, 'w', encoding='utf-8') as f:
        json.dump(risks, f, ensure_ascii=False, indent=2)
    logger.info(f"审查结果已保存至: {review_results_path}")

    # ============================================================
    # 完成
    # ============================================================
    logger.info("=" * 60)
    logger.info("全部任务完成！")
    logger.info(f"QA 结果: {qa_results_path}")
    logger.info(f"审查结果: {review_results_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
