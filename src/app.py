"""Streamlit Web — 合同 Multi-Agent 系统

布局：
- 左侧：合同库（上传 / 切换 / 重命名 / 删除）
- 主区：4 个 Tab
    Tab 1 💬 交互式问答（输入需求 → IntentRouter → Planner → Agent 流式执行 → 回答）
    Tab 2 🔁 Agent Trace（流程图 + 时间线）
    Tab 3 📋 历史结果（QA / 风险 / Diff，复用合同长期记忆）
    Tab 4 🔬 文档浏览（chunks 浏览 / 检索 demo）
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st
from dotenv import load_dotenv

from agents import SharedState
from agents.langgraph_orchestrator import LangGraphOrchestrator
from chunker import Chunk
from contract_library import ContractLibrary
from llm_client import get_default_model, get_lite_model

load_dotenv()
logging.basicConfig(level=logging.WARNING)


# ============================================================
# 页面配置 / 全局状态
# ============================================================
st.set_page_config(
    page_title="合同 Multi-Agent 系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "current_contract_id" not in st.session_state:
    st.session_state.current_contract_id = None
if "v2_contract_id" not in st.session_state:
    st.session_state.v2_contract_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # [(role, msg, agent_trace, results)]


# ============================================================
# 工具函数
# ============================================================
@st.cache_resource
def get_library() -> ContractLibrary:
    return ContractLibrary()


def load_chunks_from_path(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(chunk_id=c["chunk_id"], content=c["content"], metadata=c.get("metadata", {}))
        for c in raw
    ]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 渲染辅助
# ============================================================
SEVERITY_COLOR = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def render_citation(c: dict, chunks_by_id: dict[str, Chunk]):
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
        chunk_obj = chunks_by_id.get(chunk_id)
        if chunk_obj:
            st.markdown("**完整 chunk 原文**")
            st.code(chunk_obj.content, language="markdown")
        if c.get("reason"):
            st.markdown(f"**为什么用此引用**：{c['reason']}")


def render_risk(r: dict, chunks_by_id):
    sev = r.get("severity", "medium")
    sev_icon = SEVERITY_COLOR.get(sev, "⚪")
    title = f"{sev_icon} **{r.get('risk_id','')}** · {r.get('risk_type','')} · {r.get('title','')}"
    with st.expander(title):
        c1, c2, c3 = st.columns(3)
        c1.metric("严重度", sev.upper())
        c2.metric("置信度", f"{r.get('confidence', 0):.2f}")
        c3.metric("人工复核", "是" if r.get("needs_human_review") else "否")
        if r.get("reason"):
            st.markdown("**风险描述**")
            st.write(r["reason"])
        if r.get("suggestion"):
            st.markdown("**修改建议**")
            st.success(r["suggestion"])
        st.markdown("**支撑证据**")
        for ev in r.get("evidence", []):
            render_citation(ev, chunks_by_id)


def render_diff(d: dict):
    impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d.get("impact", "medium"), "⚪")
    diff_type = d.get("diff_type", "?")
    type_icon = {
        "changed": "🔁",
        "added": "➕",
        "removed": "➖",
        "added_section": "📄➕",
        "removed_section": "📄➖",
    }.get(diff_type, "❓")
    title = f"{impact_icon} {type_icon} {d.get('diff_id','?')} · {d.get('topic','')}"
    with st.expander(title):
        st.caption(f"section: {d.get('section', '')}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**v1（旧版）**")
            st.warning(d.get("v1_quote") or "*(无)*")
        with c2:
            st.markdown("**v2（新版）**")
            st.success(d.get("v2_quote") or "*(无)*")
        st.markdown(f"**说明**：{d.get('summary','')}")
        st.caption(
            f"impact={d.get('impact','?')} · 需人工复核：{'是' if d.get('needs_human_review') else '否'}"
        )


# ============================================================
# 侧边栏：合同库
# ============================================================
def render_sidebar() -> Optional[str]:
    """返回当前选中的 contract_id"""
    lib = get_library()

    with st.sidebar:
        st.markdown("### ⚙️ 系统配置")
        backend = "AWS Bedrock" if os.environ.get("USE_BEDROCK", "").lower() in ("1", "true", "yes") else "Anthropic"
        st.caption(f"后端：`{backend}` · 主：`{get_default_model().split('.')[-1]}` · 轻：`{get_lite_model().split('.')[-1]}`")

        st.divider()
        st.markdown("### 📚 合同库")

        # ----- 上传 -----
        with st.expander("📎 上传新合同", expanded=len(lib.list()) == 0):
            uploaded = st.file_uploader(
                "选择 PDF",
                type=["pdf"],
                accept_multiple_files=False,
                key="upload_pdf",
            )
            role = st.radio(
                "用途",
                options=["primary", "v2"],
                format_func=lambda x: "主合同" if x == "primary" else "对比合同 (v2)",
                horizontal=True,
                key="upload_role",
            )
            if uploaded is not None:
                if st.button("✅ 加入合同库", use_container_width=True):
                    info = lib.add_pdf(
                        uploaded.getvalue(),
                        original_filename=uploaded.name,
                        role=role,
                    )
                    st.success(f"已入库：{info.alias}")
                    if role == "primary":
                        st.session_state.current_contract_id = info.id
                    else:
                        st.session_state.v2_contract_id = info.id
                    st.rerun()

        # ----- 列表 -----
        contracts = lib.list()
        if not contracts:
            st.info("合同库为空，请先上传 PDF。")
            return None

        st.markdown(f"**共 {len(contracts)} 份**")
        cur = st.session_state.current_contract_id
        for c in contracts:
            is_current = c.id == cur
            badge = "✅" if is_current else "⚪"
            role_tag = "🅿️" if c.role == "primary" else "🅑"
            short = c.alias[:24] + ("…" if len(c.alias) > 24 else "")
            with st.container(border=True):
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    st.markdown(f"{badge} {role_tag} **{short}**")
                    st.caption(
                        f"`{c.id}` · {c.pages or '?'} 页 · {c.chunks or '?'} chunks · "
                        f"{c.last_accessed[:10] if c.last_accessed else '-'}"
                    )
                with col_b:
                    if not is_current and st.button("→", key=f"sel_{c.id}", help="切换到该合同"):
                        st.session_state.current_contract_id = c.id
                        lib.touch(c.id)
                        st.rerun()

        # ----- 当前合同操作 -----
        if cur:
            cur_info = lib.get(cur)
            if cur_info:
                st.divider()
                st.markdown("### 📌 当前合同")
                with st.form("rename_form", clear_on_submit=False):
                    new_alias = st.text_input("别名", value=cur_info.alias)
                    if st.form_submit_button("✏️ 重命名"):
                        lib.rename(cur, new_alias)
                        st.rerun()

                # v2 选择（用于 diff）
                v2_options = ["（无）"] + [
                    f"{c.alias} [{c.id[:14]}]"
                    for c in contracts if c.id != cur
                ]
                v2_idx = 0
                if st.session_state.v2_contract_id:
                    for i, c in enumerate([cc for cc in contracts if cc.id != cur]):
                        if c.id == st.session_state.v2_contract_id:
                            v2_idx = i + 1
                            break
                v2_pick = st.selectbox(
                    "🅑 用于对比的 v2 合同",
                    options=v2_options,
                    index=v2_idx,
                    help="选择后，问『对比新旧合同』将自动启用 DiffAgent",
                )
                if v2_pick == "（无）":
                    st.session_state.v2_contract_id = None
                else:
                    others = [c for c in contracts if c.id != cur]
                    st.session_state.v2_contract_id = others[v2_options.index(v2_pick) - 1].id

                if st.button("🗑️ 删除当前合同", type="secondary"):
                    lib.delete(cur)
                    st.session_state.current_contract_id = None
                    st.rerun()

        return cur


# ============================================================
# Tab 1：交互式问答（流式 agent 反馈）
# ============================================================
AGENT_NAME_CN = {
    "intent_router": "🎯 意图分类",
    "planner": "🧠 计划",
    "parser": "📄 解析",
    "indexer": "🔍 索引",
    "qa": "💬 问答",
    "audit": "⚠️ 审查",
    "diff": "🔀 对比",
    "reflection": "🪞 反思",
}


def render_tab_chat(contract_id: Optional[str]):
    if not contract_id:
        st.info("请先在左侧选择或上传合同。")
        return

    lib = get_library()
    info = lib.get(contract_id)
    if info is None:
        st.error("找不到合同信息")
        return

    st.markdown(f"### 💬 与「{info.alias}」对话")

    # 历史
    for entry in st.session_state.chat_history:
        role, content = entry["role"], entry["content"]
        with st.chat_message(role):
            st.markdown(content)

    # 输入框
    user_input = st.chat_input("提问任何问题（系统会自动分流：合同问题走 multi-agent，闲聊走 Haiku）...")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 跑 agent（流式）
    with st.chat_message("assistant"):
        run_agent_pipeline_streaming(user_input, contract_id, info)


def run_agent_pipeline_streaming(user_input: str, contract_id: str, info):
    """用 LangGraph 流式 invoke 实时显示每个 agent 的状态。"""
    state = SharedState(
        contract_id=contract_id,
        contract_id_v2=st.session_state.v2_contract_id,
        user_request=user_input,
        output_dir="outputs",
    )

    # 实时状态面板
    status_panel = st.status("🚀 启动 Agent 工作流...", expanded=True)
    progress_lines = []

    def render_progress():
        with status_panel:
            for line in progress_lines:
                st.markdown(line)

    orch = LangGraphOrchestrator(use_planner=True)
    final_state = state
    completed_agents = set()

    try:
        # graph.stream() 返回每个节点完成后的 state 增量
        for event in orch.graph.stream(state, config={"recursion_limit": 50}):
            # event 形如 {"node_name": <state>}
            for node_name, node_state in event.items():
                if node_name in completed_agents:
                    # 节点重跑（reflection 触发的 audit 重跑）
                    label = f"🔄 {AGENT_NAME_CN.get(node_name, node_name)}（重跑）"
                else:
                    label = f"✅ {AGENT_NAME_CN.get(node_name, node_name)}"
                completed_agents.add(node_name)

                # 把 messages 里属于该 agent 的最新那条作为子说明
                last_msg = ""
                msgs = getattr(node_state, "messages", None) or (
                    node_state.get("messages", []) if isinstance(node_state, dict) else []
                )
                if msgs:
                    for m in reversed(msgs):
                        agent_field = m.agent if hasattr(m, "agent") else m.get("agent")
                        if agent_field == node_name:
                            text = m.msg if hasattr(m, "msg") else m.get("msg", "")
                            last_msg = text
                            break
                if last_msg:
                    progress_lines.append(f"- {label}：{last_msg}")
                else:
                    progress_lines.append(f"- {label}")
                render_progress()

                # 更新 final_state
                if isinstance(node_state, dict):
                    for k, v in node_state.items():
                        if hasattr(state, k):
                            setattr(state, k, v)
                else:
                    final_state = node_state

        status_panel.update(label="✅ 工作流完成", state="complete", expanded=False)
    except Exception as e:
        status_panel.update(label=f"❌ 出错：{e}", state="error")
        st.exception(e)
        return

    # 因为 LangGraph 0.6 stream 返回 dict 增量，最终 state 已经在 state 上累积
    final_state = state

    # ---- 渲染回答 ----
    if final_state.intent == "off_topic":
        st.markdown(final_state.lite_reply)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_state.lite_reply,
        })
        return

    # contract_related：根据 plan 里实际跑过的 agent 显示对应结果
    answer_md = render_agent_results(final_state, info)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_md,
    })


def render_agent_results(state: SharedState, info) -> str:
    """渲染 agent 跑完后的结果。返回拼出来的 markdown 字符串供历史回看。"""
    md_parts = []

    # Plan 摘要
    if state.plan:
        plan_str = " → ".join(state.plan)
        md_parts.append(f"**执行计划**：{plan_str}")
        if state.plan_reasoning:
            md_parts.append(f"_Planner 解释：{state.plan_reasoning}_")
        st.info(f"**执行计划**：{plan_str}")
        if state.plan_reasoning:
            st.caption(f"💡 {state.plan_reasoning}")

    # QA 结果
    if state.qa_results:
        st.markdown("#### 💬 问答结果")
        chunks_by_id = {c.chunk_id: c for c in state.chunks}
        for q in state.qa_results:
            with st.container(border=True):
                st.markdown(f"**[{q.get('question_id','?')}]** {q.get('question','')}")
                st.write(q.get("answer", ""))
                cits = q.get("citations", [])
                if cits:
                    with st.expander(f"🔗 {len(cits)} 条引用"):
                        for c in cits:
                            render_citation(c, chunks_by_id)
                if q.get("conflicts"):
                    st.markdown("**识别到的冲突**")
                    cls_color = {"fact": "🟢", "inference": "🟡", "human_review": "🔴"}
                    for cf in q["conflicts"]:
                        st.markdown(
                            f"{cls_color.get(cf.get('conclusion_class','?'), '⚪')} **{cf.get('topic','')}** — "
                            f"{cf.get('description','')[:200]}"
                        )
        md_parts.append(f"💬 完成 {len(state.qa_results)} 个问答")

    # 风险审查
    if state.risks:
        st.markdown("#### ⚠️ 风险审查")
        sev_count = {"high": 0, "medium": 0, "low": 0}
        for r in state.risks:
            sev_count[r.get("severity", "medium")] = sev_count.get(r.get("severity", "medium"), 0) + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总数", len(state.risks))
        c2.metric("🔴 high", sev_count["high"])
        c3.metric("🟡 medium", sev_count["medium"])
        c4.metric("🔵 low", sev_count["low"])
        chunks_by_id = {c.chunk_id: c for c in state.chunks}
        for r in state.risks[:20]:
            render_risk(r, chunks_by_id)
        if len(state.risks) > 20:
            st.caption(f"…还有 {len(state.risks) - 20} 条风险，详见 Tab 3 历史结果")
        md_parts.append(f"⚠️ 识别 {len(state.risks)} 条风险（high {sev_count['high']} / medium {sev_count['medium']} / low {sev_count['low']}）")

    # Reflection
    if state.reflection_notes:
        st.markdown("#### 🪞 反思笔记")
        for n in state.reflection_notes:
            st.markdown(f"- {n}")
        md_parts.append(f"🪞 反思 {len(state.reflection_notes)} 条")

    # Diff
    if state.diff_results:
        st.markdown("#### 🔀 跨合同对比")
        st.metric("差异数", len(state.diff_results))
        for d in state.diff_results[:20]:
            render_diff(d)
        md_parts.append(f"🔀 跨合同对比 {len(state.diff_results)} 条差异")

    if not md_parts:
        return "（流水线已执行，但未产出结果）"
    return "\n\n".join(md_parts)


# ============================================================
# Tab 2：Agent Trace 可视化
# ============================================================
def render_tab_trace(contract_id: Optional[str]):
    st.markdown("### 🔁 Agent Trace")
    st.caption("LangGraph multi-agent 工作流的可视化流程图（手写清晰版）。")

    mmd_path = Path("docs/agent_workflow.mmd")
    if mmd_path.exists():
        mermaid_text = mmd_path.read_text(encoding="utf-8")
        # 去掉 mermaid 头部的 yaml meta（streamlit 不识别）
        if mermaid_text.startswith("---"):
            parts = mermaid_text.split("---", 2)
            if len(parts) >= 3:
                mermaid_text = parts[2].strip()
        st.markdown(f"```mermaid\n{mermaid_text}\n```")

    st.divider()
    st.markdown("#### ⏱️ 最近一次运行时间线")
    trace_path = Path("outputs/agent_trace.json")
    if not trace_path.exists():
        st.info("尚无 trace 数据，先在 Tab 1 跑一次 agent。")
        return

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not trace:
        st.info("trace 为空。")
        return

    for m in trace:
        ts = m.get("timestamp", "")
        agent = m.get("agent", "?")
        level = m.get("level", "info")
        msg = m.get("msg", "")
        icon = {"info": "✓", "warning": "⚠️", "error": "❌"}.get(level, "·")
        cn_name = AGENT_NAME_CN.get(agent, agent)
        st.markdown(f"`{ts[-8:]}` {icon} **{cn_name}**：{msg}")


# ============================================================
# Tab 3：历史结果（从 ContractLibrary 长期记忆读取）
# ============================================================
def render_tab_results(contract_id: Optional[str]):
    if not contract_id:
        st.info("请先选择或上传合同。")
        return
    lib = get_library()
    info = lib.get(contract_id)
    paths = lib.paths(contract_id)

    st.markdown(f"### 📋 「{info.alias}」的历史结果")
    st.caption(f"来自长期记忆：`{paths['base']}`")

    chunks = load_chunks_from_path(paths["chunks"])
    chunks_by_id = {c.chunk_id: c for c in chunks}

    qa = load_json(paths["qa_results"]) or []
    risks = load_json(paths["review_results"]) or []
    diffs = load_json(paths["diff_results"]) or []

    sub1, sub2, sub3 = st.tabs([
        f"💬 问答 ({len(qa)})",
        f"⚠️ 风险 ({len(risks)})",
        f"🔀 对比 ({len(diffs)})",
    ])
    with sub1:
        if not qa:
            st.info("暂无 QA 历史。")
        for q in qa:
            with st.container(border=True):
                st.markdown(f"**[{q.get('question_id','?')}]** {q.get('question','')}")
                st.write(q.get("answer", ""))
                cits = q.get("citations", [])
                if cits:
                    with st.expander(f"🔗 {len(cits)} 条引用"):
                        for c in cits:
                            render_citation(c, chunks_by_id)
                if q.get("conflicts"):
                    st.markdown("**冲突**")
                    for cf in q["conflicts"]:
                        st.caption(f"- [{cf.get('conclusion_class','?')}] {cf.get('topic','')}: {cf.get('description','')[:120]}")
    with sub2:
        if not risks:
            st.info("暂无审查历史。")
        else:
            sev_filter = st.multiselect("过滤严重度", ["high", "medium", "low"], default=["high", "medium", "low"])
            for r in risks:
                if r.get("severity") in sev_filter:
                    render_risk(r, chunks_by_id)
    with sub3:
        if not diffs:
            st.info("暂无对比历史（在 Tab 1 选择 v2 合同并问『对比』）。")
        else:
            for d in diffs:
                render_diff(d)


# ============================================================
# Tab 4：文档浏览
# ============================================================
def render_tab_explore(contract_id: Optional[str]):
    if not contract_id:
        st.info("请先选择或上传合同。")
        return
    lib = get_library()
    paths = lib.paths(contract_id)
    chunks = load_chunks_from_path(paths["chunks"])
    if not chunks:
        st.info("尚未生成 chunks，先到 Tab 1 提问触发解析与索引。")
        return

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
    for c in filtered[:50]:
        m = c.metadata
        head = (
            f"`{c.chunk_id}` · {m.get('block_type', '')} · "
            f"第 {m.get('page_hint', '?')} 页 · {m.get('section_path', '')[:50]}"
        )
        with st.expander(head):
            st.code(c.content, language="markdown")


# ============================================================
# 主入口
# ============================================================
st.title("📄 合同 Multi-Agent 系统")
st.caption(
    "上传合同 PDF → 提问任意需求 → IntentRouter 分流 → "
    "PlannerAgent 决策 → 7 个 Agent 协作完成（流式实时反馈）"
)

contract_id = render_sidebar()

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 交互问答",
    "🔁 Agent Trace",
    "📋 历史结果",
    "🔬 文档浏览",
])
with tab1:
    render_tab_chat(contract_id)
with tab2:
    render_tab_trace(contract_id)
with tab3:
    render_tab_results(contract_id)
with tab4:
    render_tab_explore(contract_id)
