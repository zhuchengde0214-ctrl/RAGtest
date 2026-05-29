"""
Evaluation 脚本

针对 evals/golden.jsonl 中手工标注的 10 个 QA，评估三个维度：

1. Retrieval Recall@K（检索召回）
   - 给定问题，看 top-K 检索结果中是否包含 expected_section_substring 所在的 chunk
2. Citation Hit Rate（引用命中率）
   - 调 LLM 生成答案后，看 citations 里是否真正包含 expected_section_substring 所在的 chunk
3. Answer Keyword Coverage（关键词覆盖）
   - 答案中是否出现 expected_answer_keywords 列表的关键词

输出：
- 控制台 summary
- evals/eval_report.json（详细每条结果）
- docs/evaluation.md（人类可读报告）

用法：
    python3 src/eval.py                # 跑全部
    python3 src/eval.py --no-llm       # 仅算 retrieval recall（不调 LLM）
    python3 src/eval.py --top-k 5      # 自定义 top_k
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

from pdf_parser import load_parsed_document
from chunker import DocumentChunker
from retriever import EmbeddingProvider, Retriever
from qa_engine import QAEngine
from llm_client import get_default_model

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
EVALS = ROOT / "evals"
DOCS = ROOT / "docs"


def load_golden():
    items = []
    with open(EVALS / "golden.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_pipeline():
    parsed_path = OUTPUTS / "parsed_document.json"
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"未找到 {parsed_path}，请先跑 `python3 src/main.py` 生成基线产物。"
        )
    parsed = load_parsed_document(str(parsed_path))
    chunks = DocumentChunker().chunk(parsed)

    use_bedrock_emb = os.environ.get("USE_BEDROCK_EMBEDDING", "").lower() in ("1", "true", "yes")
    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    embedding = EmbeddingProvider(
        use_bedrock=use_bedrock_emb,
        use_local=use_local,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    retriever = Retriever(
        embedding_provider=embedding,
        persist_dir=str(OUTPUTS / "chroma_db_eval"),
        vector_top_k=int(os.environ.get("VECTOR_TOP_K", "10")),
        bm25_top_k=int(os.environ.get("BM25_TOP_K", "10")),
        rerank_top_k=int(os.environ.get("RERANK_TOP_K", "6")),
    )
    retriever.index(chunks)
    return parsed, chunks, retriever


# ------------------------------------------------------------------
# 三个评估指标
# ------------------------------------------------------------------
def hits_section(chunks_or_results, expected_substring: str) -> bool:
    """results 可以是 retriever.search 返回值或 citations。返回 expected_substring 是否在任意结果的 section_path 中。"""
    if not chunks_or_results:
        return False
    for r in chunks_or_results:
        meta = r.get("metadata", {}) if "metadata" in r else r
        section = meta.get("section_path", "") or r.get("section", "")
        if expected_substring in section:
            return True
    return False


def keyword_coverage(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """关键词覆盖率：命中数 / 总数。返回 (coverage, hit_list)"""
    if not keywords:
        return 1.0, []
    hits = [k for k in keywords if k in text]
    return len(hits) / len(keywords), hits


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def evaluate(top_k: int = 6, run_llm: bool = True):
    golden = load_golden()
    print(f"加载 {len(golden)} 条 golden 测试样本")
    parsed, chunks, retriever = build_pipeline()

    qa_engine = None
    if run_llm:
        qa_engine = QAEngine(model=get_default_model())

    results = []
    for g in golden:
        qid = g["id"]
        question = g["question"]
        exp_section = g.get("expected_section_substring", "")
        exp_kws = g.get("expected_answer_keywords", [])

        # --- 1. Retrieval Recall ---
        retrieved = retriever.search(question, top_k=top_k)
        recall_hit = hits_section(retrieved, exp_section)

        record = {
            "id": qid,
            "question": question,
            "expected_section": exp_section,
            "expected_keywords": exp_kws,
            "retrieval": {
                "top_k": top_k,
                "hit": recall_hit,
                "top_sections": [r["metadata"].get("section_path", "") for r in retrieved[:top_k]],
            },
        }

        # --- 2. Citation Hit + 3. Keyword Coverage ---
        if run_llm and qa_engine:
            try:
                ans = qa_engine.answer_simple(question, retriever, top_k=top_k)
                citations = ans.get("citations", [])
                citation_hit = hits_section(citations, exp_section)
                coverage, hit_kws = keyword_coverage(ans.get("answer", ""), exp_kws)

                record["llm"] = {
                    "answer": ans.get("answer", ""),
                    "confidence": ans.get("confidence"),
                    "citation_hit": citation_hit,
                    "n_citations": len(citations),
                    "n_resolved": sum(1 for c in citations if c.get("resolved")),
                    "keyword_coverage": coverage,
                    "hit_keywords": hit_kws,
                    "missed_keywords": [k for k in exp_kws if k not in hit_kws],
                }
            except Exception as e:
                logger.warning(f"{qid} LLM 调用失败: {e}")
                record["llm"] = {"error": str(e)}

        results.append(record)
        # 实时进度
        marks = []
        marks.append("R✓" if recall_hit else "R✗")
        if "llm" in record and "citation_hit" in record["llm"]:
            marks.append("C✓" if record["llm"]["citation_hit"] else "C✗")
            marks.append(f"K{record['llm']['keyword_coverage']:.0%}")
        print(f"  {qid} {' '.join(marks):14s} {question[:40]}")

    return results


def summarize(results):
    n = len(results)
    recall_hit = sum(1 for r in results if r["retrieval"]["hit"])
    citation_hit = sum(1 for r in results if r.get("llm", {}).get("citation_hit"))
    avg_kw = (
        sum(r.get("llm", {}).get("keyword_coverage", 0) for r in results) / n
        if any("llm" in r for r in results)
        else None
    )
    avg_resolved_rate = (
        sum(
            (r["llm"]["n_resolved"] / r["llm"]["n_citations"])
            if r.get("llm", {}).get("n_citations")
            else 0
            for r in results
            if "llm" in r and "n_citations" in r["llm"]
        )
        / n
        if any("llm" in r and "n_citations" in r["llm"] for r in results)
        else None
    )

    return {
        "n_samples": n,
        "retrieval_recall": recall_hit / n,
        "citation_hit_rate": citation_hit / n if any("llm" in r for r in results) else None,
        "avg_keyword_coverage": avg_kw,
        "avg_resolved_rate": avg_resolved_rate,
    }


def write_report(results, summary, top_k: int):
    EVALS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    # JSON 详细
    with open(EVALS / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results, "top_k": top_k},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Markdown 报告
    lines = []
    lines.append("# Evaluation 报告\n")
    lines.append("基于 `evals/golden.jsonl` 的 10 条手工标注 QA 自动评估。\n")
    lines.append("## 总体指标\n")
    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("|---|---:|---|")
    lines.append(f"| 样本数 | {summary['n_samples']} | golden 集大小 |")
    lines.append(
        f"| Retrieval Recall@{top_k} | "
        f"{summary['retrieval_recall']:.0%} | "
        f"top-{top_k} 检索结果至少有一个命中预期章节 |"
    )
    if summary["citation_hit_rate"] is not None:
        lines.append(
            f"| Citation Hit Rate | "
            f"{summary['citation_hit_rate']:.0%} | "
            f"LLM 生成的 citations 至少有一个命中预期章节 |"
        )
    if summary["avg_keyword_coverage"] is not None:
        lines.append(
            f"| 关键词覆盖率（平均） | "
            f"{summary['avg_keyword_coverage']:.0%} | "
            f"答案中出现的预期关键词比例 |"
        )
    if summary["avg_resolved_rate"] is not None:
        lines.append(
            f"| 引用回链成功率 | "
            f"{summary['avg_resolved_rate']:.0%} | "
            f"LLM 输出的 chunk_id 直接命中真实 chunk 的比例 |"
        )

    lines.append("\n## 逐条结果\n")
    for r in results:
        retrieval = r["retrieval"]
        marks = ["✅" if retrieval["hit"] else "❌"]
        kw = ""
        cit = ""
        ans_text = ""
        if "llm" in r and "citation_hit" in r["llm"]:
            marks.append("✅" if r["llm"]["citation_hit"] else "❌")
            kw = f"关键词覆盖：{r['llm']['keyword_coverage']:.0%}"
            cit = (
                f"引用 {r['llm']['n_citations']} 条，"
                f"resolved {r['llm']['n_resolved']} 条"
            )
            ans_text = r["llm"].get("answer", "")[:200]
        lines.append(
            f"### {r['id']} {' '.join(marks)} `{r['question']}`\n"
        )
        lines.append(f"- **预期章节**：包含 `{r['expected_section']}`")
        lines.append(
            f"- **检索 top sections**：{', '.join(retrieval['top_sections'][:3])}"
        )
        if kw:
            lines.append(f"- **{kw}**")
            if r["llm"].get("missed_keywords"):
                lines.append(
                    f"  - 未命中关键词：{', '.join(r['llm']['missed_keywords'])}"
                )
        if cit:
            lines.append(f"- **{cit}**")
        if ans_text:
            lines.append(f"- **答案**：{ans_text}…")
        lines.append("")

    (DOCS / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📝 报告已生成：{DOCS/'evaluation.md'}")
    print(f"📊 详细数据：{EVALS/'eval_report.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--no-llm", action="store_true", help="只算 retrieval recall，不调 LLM")
    args = p.parse_args()

    results = evaluate(top_k=args.top_k, run_llm=not args.no_llm)
    summary = summarize(results)

    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"样本数:                {summary['n_samples']}")
    print(f"Retrieval Recall@{args.top_k}:    {summary['retrieval_recall']:.0%}")
    if summary["citation_hit_rate"] is not None:
        print(f"Citation Hit Rate:     {summary['citation_hit_rate']:.0%}")
    if summary["avg_keyword_coverage"] is not None:
        print(f"关键词覆盖率（平均）:    {summary['avg_keyword_coverage']:.0%}")
    if summary["avg_resolved_rate"] is not None:
        print(f"引用回链成功率:          {summary['avg_resolved_rate']:.0%}")

    write_report(results, summary, args.top_k)


if __name__ == "__main__":
    main()
