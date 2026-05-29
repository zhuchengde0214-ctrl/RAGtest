# 系统架构

## 整体分层

```
┌─────────────────────────────────────────────────────────────────┐
│  应用层（CLI / Streamlit Web / Eval）                            │
│   src/main.py · src/app.py · src/eval.py                         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  编排层（Multi-Agent Orchestration）                             │
│   ┌──────────────────────────┐    ┌──────────────────────────┐  │
│   │ Orchestrator (自研)      │    │ LangGraphOrchestrator    │  │
│   │ - 顺序执行 plan          │    │ - StateGraph + 条件边    │  │
│   │ - ReAct 循环手写         │    │ - 自动 mermaid 流程图    │  │
│   └──────────────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 层（src/agents/）                                         │
│   BaseAgent: invoke() 统一入口（日志 + 计时 + 异常捕获）         │
│   ┌─────────┬──────────┬──────────┬──────────┬──────────┐       │
│   │ Planner │ Parser   │ Indexer  │ QA       │ Audit    │       │
│   ├─────────┼──────────┼──────────┼──────────┼──────────┤       │
│   │ Diff    │ Reflect  │          │          │          │       │
│   └─────────┴──────────┴──────────┴──────────┴──────────┘       │
│   共享 SharedState（dataclass）：parsed_doc / chunks / risks /  │
│   plan / messages / needs_rerun ...                              │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  能力层（src/）                                                  │
│   pdf_parser.py  Vision OCR + 跨页表格合并                       │
│   chunker.py     混合分块（block 分组 + 递归切分 + overlap）     │
│   retriever.py   Hybrid（BM25+jieba+2gram + Chroma + RRF）       │
│                  + locate_evidence 证据回链                      │
│   qa_engine.py   三类问答 + JSON 修复（_escape_inner_quotes /    │
│                  _repair_truncated_json）                        │
│   review_engine.py  10 维度审查（每维度独立检索独立调用）        │
│   llm_client.py  Anthropic / Bedrock 双后端工厂                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  外部依赖                                                         │
│   Claude Sonnet (Vision + Text)  ChromaDB  SentenceTransformer   │
│   jieba  rank_bm25  PyMuPDF                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 多 Agent 工作流（plan 执行举例）

### 用户请求 1："只想问问题"
```
PlannerAgent → plan = [parser, indexer, qa]
  → ParserAgent (Vision OCR)
  → IndexerAgent (chunk + index)
  → QAAgent (Q1/Q2/Q3)
END
```

### 用户请求 2："帮我做合规审查，重点关注金额一致性"
```
PlannerAgent → plan = [parser, indexer, audit, reflection]
  → ParserAgent
  → IndexerAgent
  → AuditAgent (10 维度，state.user_request 提示重点)
  → ReflectionAgent
       └─ needs_rerun=True → AuditAgent (用 reflection_notes 增强 prompt)
       └─ ReflectionAgent (max_iters=1，直接放行)
END
```

### 用户请求 3："对比新旧合同"
```
PlannerAgent → plan = [parser, indexer, diff]
  → ParserAgent (同时解析 v1 + v2)
  → IndexerAgent (建两个索引)
  → DiffAgent (章节对齐 + LLM 条款级对比 + chunk 回链)
END
```

## SharedState 关键字段

```python
@dataclass
class SharedState:
    # 输入
    pdf_path: str
    pdf_path_v2: Optional[str]
    user_request: str
    output_dir: str

    # 各 agent 产物
    plan: list[str]                # PlannerAgent
    plan_reasoning: str

    parsed_doc: ParsedDocument     # ParserAgent
    parsed_doc_v2: Optional[...]

    chunks: list[Chunk]            # IndexerAgent
    chunks_v2: list
    retriever: Retriever
    retriever_v2: Optional[...]

    qa_results: list[dict]         # QAAgent
    risks: list[dict]              # AuditAgent
    diff_results: list[dict]       # DiffAgent

    reflection_notes: list[str]    # ReflectionAgent
    reflection_iters: int
    needs_rerun: Optional[str]     # 触发 ReAct

    # 全局
    messages: list[AgentMessage]   # 全部 agent 的 invoke 日志
    errors: list[str]
```

## ReAct 循环细节

```
    [audit] → [reflection]
                  │
                  ├─ needs_rerun=audit & iters<MAX → [audit] (带 reflection_notes 重跑)
                  └─ otherwise → END

    audit 第二次执行时，prompt 头部会被注入：
       ## 上一轮审视发现的问题（请本轮重点关注）
       - 可能遗漏：知识产权归属条款缺失
       - 严重度不一致：R025 / R027 应升为 high
       - ...
```

## 设计权衡

| 决策 | 选 A | 选 B（不选） | 理由 |
|---|---|---|---|
| 编排框架 | LangGraph + 自研 BaseAgent 混合 | 纯 LangGraph 节点函数 / 纯自研 | 框架光环 + 节点逻辑可控；面试可清晰讲清每一层 |
| Agent 状态 | 单 SharedState dataclass | 每 agent 独立 in/out | dataclass 直接给 LangGraph state schema，状态流向清晰 |
| ReAct 实现 | reflection 写 needs_rerun，编排器看字段决定 | 让 LLM 直接调用工具 | 当前所有 agent 都内部带 LLM 调用，工具 = agent；reflection 决策由 LLM 出，路由由代码做 |
| LLM 后端 | Anthropic / Bedrock 工厂模式 | 硬编码某一个 | 评审环境多样；面试讲"对接 AWS Bedrock 业务现实" |
| Embedding | 本地 sentence-transformers 默认 | 必须 OpenAI | 评审/CI 环境无 OpenAI key 也能跑；BM25+2gram 兜底 |
| 引用回链 | LLM 输出 chunk_id + quote 双路兜底 | 只让 LLM 引用（信任 chunk_id） | LLM 偶尔写错 chunk_id；quote 子串/BM25 反查保证 100% 可定位 |
