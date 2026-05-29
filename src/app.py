"""
Streamlit Web Demo — 合同 RAG 问答与风险审查

用法：
    streamlit run src/app.py

依赖项目内的：pdf_parser / chunker / retriever / qa_engine / review_engine
首次运行会读取 outputs/parsed_document.json 和 outputs/chunks.json（如已存在），
否则提示用户先跑 main.py 生成基线产物。
"""

import json
import os
import sys
from pathlib import Path

# sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
from dotenv import load_dotenv

from pdf_parser import load_parsed_document
from chunker import DocumentChunker
from retriever import EmbeddingProvider, Retriever
from qa_engine import QAEngine
from review_engine import ReviewEngine
from llm_client import get_default_model

load_dotenv()


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="合同 RAG 问答与审查",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 缓存：避免每次交互都重建索引
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


@st.cache_resource(show_spinner="正在加载文档与索引...")
def get_pipeline():
    """加载已落盘的 parsed_document.json + 重建索引。返回 (parsed, chunks, retriever, qa, review)。"""
    parsed_path = OUTPUTS / "parsed_document.json"
    if not parsed_path.exists():
        return None

    parsed = load_parsed_document(str(parsed_path))
    chunks = DocumentChunker().chunk(parsed)

    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    embedding = EmbeddingProvider(
        use_local=use_local,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
    retriever = Retriever(
        embedding_provider=embedding,
        persist_dir=str(OUTPUTS / "chroma_db"),
        vector_top_k=int(os.environ.get("VECTOR_TOP_K", "10")),
        bm25_top_k=int(os.environ.get("BM25_TOP_K", "10")),
        rerank_top_k=int(os.environ.get("RERANK_TOP_K", "6")),
    )
    retriever.index(chunks)

    qa = QAEngine(model=get_default_model())
    review = ReviewEngine(model=get_default_model())

    return parsed, chunks, retriever, qa, review


def load_existing_qa():
    p = OUTPUTS / "qa_results.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return None


def load_existing_review():
    p = OUTPUTS / "review_results.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return None


# ============================================================
# Helpers
# ============================================================
def chunk_lookup(chunks, chunk_id: str):
    for c in chunks:
        if c.chunk_id == chunk_id:
            return c
    return None


def render_citation(c: dict, chunks):
    """渲染一条引用，可点击展开看原文。"""
    sec = c.get("section") or "(未知章节)"
    pages = c.get("pages")
    if isinstance(pages, list) and pages:
        page_str = "-".join(str(p) for p in pages) if len(pages) > 1 else str(pages[0])
    elif pages:
        page_str = str(pages)
    elif c.get("page_hint"):
        page_str = str(c["page_hint"])
    else:
        page_str = "?"
    chunk_id = c.get("source_id", "")
    quote = c.get("quote", "")
    block_type = c.get("block_type", "")
    resolved = c.get("resolved", True)

    badge = "🟢" if resolved else "🟡"
    title = f"{badge} `{chunk_id}` · 第 {page_str} 页 · {block_type} · {sec[:40]}"

    with st.expander(title):
        if quote:
            st.markdown("**引用片段**")
            st.info(quote)
        chunk_obj = chunk_lookup(chunks, chunk_id)
        if chunk_obj:
            st.markdown("**完整 chunk 原文**")
            st.code(chunk_obj.content, language="markdown")
            st.caption(f"chunk_index={chunk_obj.metadata.get('chunk_index')} · "
                       f"char_len={chunk_obj.metadata.get('char_len')}")
        if c.get("reason"):
            st.markdown(f"**为什么用此引用**：{c['reason']}")


SEVERITY_COLOR = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def render_risk(r: dict, chunks):
    sev = r.get("severity", "medium")
    sev_icon = SEVERITY_COLOR.get(sev, "⚪")
    title = f"{sev_icon} **{r.get('risk_id', '')}** · {r.get('risk_type', '')} · {r.get('title', '')}"
    needs_review = r.get("needs_human_review")

    with st.expander(title):
        cols = st.columns([1, 1, 1])
        cols[0].metric("严重度", sev.upper())
        cols[1].metric("置信度", f"{r.get('confidence', 0):.2f}")
        cols[2].metric("人工复核", "是" if needs_review else "否")

        if r.get("reason"):
            st.markdown("**风险描述**")
            st.write(r["reason"])
        if r.get("suggestion"):
            st.markdown("**修改建议**")
            st.success(r["suggestion"])

        st.markdown("**支撑证据**")
        for ev in r.get("evidence", []):
            render_citation(ev, chunks)


# ============================================================
# UI
# ============================================================
st.title("📄 合同 RAG 问答与风险审查")
st.caption("基于扫描件 PDF 的端到端 RAG 流水线：Vision OCR · Hybrid 检索 · 多轮 QA · 10 维度风险审查")

with st.sidebar:
    st.header("⚙️ 配置")
    backend = "AWS Bedrock" if os.environ.get("USE_BEDROCK", "").lower() in ("1", "true", "yes") else "Anthropic API"
    st.markdown(f"**LLM 后端**：`{backend}`")
    st.markdown(f"**模型**：`{get_default_model()}`")

    st.divider()
    st.header("📊 项目状态")
    parsed_exists = (OUTPUTS / "parsed_document.json").exists()
    chunks_exists = (OUTPUTS / "chunks.json").exists()
    qa_exists = (OUTPUTS / "qa_results.json").exists()
    review_exists = (OUTPUTS / "review_results.json").exists()

    st.markdown(f"- {'✅' if parsed_exists else '❌'} 文档解析 (parsed_document.json)")
    st.markdown(f"- {'✅' if chunks_exists else '❌'} 分块结果 (chunks.json)")
    st.markdown(f"- {'✅' if qa_exists else '❌'} QA 基线 (qa_results.json)")
    st.markdown(f"- {'✅' if review_exists else '❌'} 审查基线 (review_results.json)")

    if not parsed_exists:
        st.warning("尚未生成解析结果。请先运行：\n```bash\npython3 src/main.py\n```")
        st.stop()

    st.divider()
    st.header("🛠️ 操作")
    if st.button("🔄 重建检索索引"):
        st.cache_resource.clear()
        st.rerun()

# 加载流水线
pipeline = get_pipeline()
if pipeline is None:
    st.error("无法加载文档解析结果，请先运行 `python3 src/main.py`。")
    st.stop()

parsed, chunks, retriever, qa_engine, review_engine = pipeline

# 顶部统计
col1, col2, col3, col4 = st.columns(4)
col1.metric("PDF 页数", parsed.total_pages)
col2.metric("解析 block 数", len(parsed.blocks))
col3.metric("索引 chunk 数", len(chunks))
col4.metric("OCR 不确定 block", sum(1 for b in parsed.blocks if b.needs_review))

st.divider()

tab_qa, tab_review, tab_explore = st.tabs(["💬 问答", "⚠️ 风险审查", "🔍 文档浏览"])


# ============================================================
# Tab 1：问答
# ============================================================
with tab_qa:
    qa_data = load_existing_qa() or []
    qa_by_id = {x["question_id"]: x for x in qa_data}

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
        ["Q1 简单事实", "Q2 多轮追问", "Q3 复杂推理", "💡 自由提问"]
    )

    # --- Q1 ---
    with sub_tab1:
        q1 = qa_by_id.get("Q1")
        if q1:
            st.markdown(f"**问题**：{q1['question']}")
            st.markdown("**回答**")
            st.write(q1["answer"])
            st.markdown(f"**置信度**：{q1.get('confidence', 0):.2f}")
            st.markdown("**引用**")
            for c in q1.get("citations", []):
                render_citation(c, chunks)
        else:
            st.info("尚无 Q1 基线结果。")

    # --- Q2 ---
    with sub_tab2:
        for tid in ("Q2-1", "Q2-2", "Q2-3"):
            item = qa_by_id.get(tid)
            if not item:
                continue
            st.markdown(f"### 第 {item.get('turn', '?')} 轮")
            st.markdown(f"**问题**：{item['question']}")
            if item.get("rewritten_question"):
                st.caption(f"改写后用于检索：{item['rewritten_question']}")
            st.write(item["answer"])
            st.markdown(f"置信度：{item.get('confidence', 0):.2f}")
            with st.expander(f"🔗 {len(item.get('citations', []))} 条引用"):
                for c in item.get("citations", []):
                    render_citation(c, chunks)
            st.divider()

    # --- Q3 ---
    with sub_tab3:
        q3 = qa_by_id.get("Q3")
        if q3:
            st.markdown(f"**问题**：{q3['question']}")
            st.markdown("**综合判断**")
            st.write(q3["answer"])
            st.markdown(f"**置信度**：{q3.get('confidence', 0):.2f}")

            conflicts = q3.get("conflicts", [])
            if conflicts:
                st.markdown(f"### 识别到 {len(conflicts)} 项冲突 / 一致性核对")
                cls_color = {"fact": "🟢", "inference": "🟡", "human_review": "🔴"}
                for cf in conflicts:
                    cls = cf.get("conclusion_class", "human_review")
                    st.markdown(
                        f"#### {cls_color.get(cls, '⚪')} {cf.get('topic', '')} "
                        f"<span style='font-size:0.8em;color:gray'>[{cls}]</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(cf.get("description", ""))
                    if cf.get("evidence"):
                        with st.expander(f"🔗 {len(cf['evidence'])} 条证据"):
                            for ev in cf["evidence"]:
                                render_citation(ev, chunks)
        else:
            st.info("尚无 Q3 基线结果。")

    # --- 自由提问 ---
    with sub_tab4:
        st.markdown("基于已建立的检索索引现场提问。第一次提问会调用 LLM，约 10-30 秒。")
        user_q = st.text_area("你的问题", placeholder="例如：本合同的付款节点和比例是怎么安排的？", height=80)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            mode = st.radio("回答模式", ["简单事实", "复杂推理"], horizontal=True)
        with col_b:
            ask = st.button("🚀 提问", type="primary", use_container_width=True)

        if ask and user_q.strip():
            with st.spinner("正在检索 + 生成答案..."):
                try:
                    if mode == "简单事实":
                        result = qa_engine.answer_simple(user_q.strip(), retriever)
                    else:
                        result = qa_engine.answer_complex(user_q.strip(), retriever)
                    st.markdown("### 回答")
                    st.write(result["answer"])
                    st.caption(f"置信度：{result.get('confidence', 0):.2f}")

                    if result.get("conflicts"):
                        st.markdown("### 识别到的冲突")
                        for cf in result["conflicts"]:
                            st.markdown(f"- **{cf.get('topic', '')}** [{cf.get('conclusion_class', '?')}]")
                            st.caption(cf.get("description", ""))

                    st.markdown("### 引用")
                    for c in result.get("citations", []):
                        render_citation(c, chunks)
                except Exception as e:
                    st.error(f"出错了：{e}")


# ============================================================
# Tab 2：风险审查
# ============================================================
with tab_review:
    risks = load_existing_review() or []
    if not risks:
        st.info("尚无审查基线结果。运行 `python3 src/main.py` 生成。")
    else:
        # 顶部统计
        sev_count = {"high": 0, "medium": 0, "low": 0}
        for r in risks:
            sev_count[r.get("severity", "medium")] = sev_count.get(r.get("severity", "medium"), 0) + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总风险数", len(risks))
        c2.metric("🔴 high", sev_count["high"])
        c3.metric("🟡 medium", sev_count["medium"])
        c4.metric("🔵 low", sev_count["low"])

        # 过滤
        col_filter1, col_filter2 = st.columns([2, 2])
        with col_filter1:
            sev_filter = st.multiselect(
                "按严重度过滤",
                options=["high", "medium", "low"],
                default=["high", "medium", "low"],
            )
        with col_filter2:
            types = sorted({r.get("risk_type", "") for r in risks})
            type_filter = st.multiselect("按类型过滤", options=types, default=types)

        st.divider()

        filtered = [
            r for r in risks
            if r.get("severity") in sev_filter and r.get("risk_type") in type_filter
        ]
        st.caption(f"显示 {len(filtered)} / {len(risks)} 条风险")

        for r in filtered:
            render_risk(r, chunks)


# ============================================================
# Tab 3：文档浏览
# ============================================================
with tab_explore:
    st.markdown("浏览所有已索引的 chunk，支持按章节和类型过滤。")

    sections = sorted({c.metadata.get("section_path", "") for c in chunks})
    types = sorted({c.metadata.get("block_type", "") for c in chunks})

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        sec_filter = st.multiselect("章节", sections, default=sections)
    with col2:
        type_filter = st.multiselect("类型", types, default=types)
    with col3:
        keyword = st.text_input("内容关键词", "")

    filtered = [
        c for c in chunks
        if c.metadata.get("section_path", "") in sec_filter
        and c.metadata.get("block_type", "") in type_filter
        and (not keyword or keyword in c.content)
    ]
    st.caption(f"显示 {len(filtered)} / {len(chunks)} 个 chunk")

    for c in filtered:
        m = c.metadata
        head = (
            f"`{c.chunk_id}` · {m.get('block_type', '')} · "
            f"第 {m.get('page_hint', '?')} 页 · {m.get('section_path', '')[:50]}"
        )
        with st.expander(head):
            st.code(c.content, language="markdown")
            st.caption(
                f"chunk_index={m.get('chunk_index')} · char_len={m.get('char_len')} · "
                f"pages={m.get('pages')} · needs_review={m.get('needs_review', False)}"
            )

st.divider()
st.caption("源码：[GitHub](https://github.com/) · 详细策略：见 docs/chunking_and_retrieval.md")
