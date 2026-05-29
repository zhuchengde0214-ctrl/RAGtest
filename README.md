# 合同 RAG 问答与风险审查系统

> 基于扫描件 PDF 的端到端 RAG 流水线：**Vision OCR → 语义分块 → 混合检索 → 多轮问答 + 10 维度风险审查**。所有引用可定位到具体章节/页码/表格。

![demo placeholder](docs/assets/demo.gif)
*(完整运行截图见 `docs/assets/`，Web 界面见 §3)*

---

## ✨ 亮点

- **真做 RAG**，不是"全文塞 LLM"：合同审查按 10 个语义维度独立检索 + 独立调用，避免 Lost-in-the-Middle
- **引用可回链**：LLM 输出 `chunk_id + quote`，错误时用子串/BM25 反查到真实 chunk —— Citation Hit Rate 见 [`docs/evaluation.md`](docs/evaluation.md)
- **中文检索增强**：`jieba` + 2-gram 字符级兜底，覆盖法律术语未登录词；BM25 + dense 用 RRF 融合
- **跨页表格合并**：报价表跨页时表头继承，避免数据行错位
- **OCR 不确定项进入人工复核**：印章遮挡用 `[?]` 占位 → `needs_review=True` → 自动产出 medium 级风险
- **JSON 输出鲁棒性**：自研 `_escape_inner_quotes` + `_repair_truncated_json`，处理 LLM 在中文字段嵌入未转义引号 / max_tokens 截断
- **AWS Bedrock / Anthropic 双后端** 一键切换；本地 SentenceTransformer 兜底，无 OpenAI key 也能跑

---

## 🚀 Quick Start

### 方式 A：Docker（推荐）

```bash
git clone <this-repo>.git && cd RAGtest

# 编辑 .env 填入凭证（Bedrock 用 IAM 凭证；或填 ANTHROPIC_API_KEY）
cp .env.example .env

# 启动 Web demo
docker compose up

# 浏览器打开 http://localhost:8501
```

### 方式 B：Python 直跑

```bash
pip install -r requirements.txt streamlit

cp .env.example .env  # 填入凭证

# 1) 跑批处理生成基线 JSON（OCR + QA + 审查）
python3 src/main.py

# 2) 启 Web demo
streamlit run src/app.py
```

---

## 📁 目录结构

```
.
├── README.md
├── Dockerfile / docker-compose.yml         一键 Docker 部署
├── .env.example                            环境变量模板
├── requirements.txt
├── run.sh / run.bat                        Linux-macOS / Windows 启动脚本
├── src/
│   ├── main.py                             批处理主流程
│   ├── app.py                              Streamlit Web demo
│   ├── eval.py                             evaluation 自动化
│   ├── pdf_parser.py                       Vision OCR + 跨页表格合并
│   ├── chunker.py                          语义分块（block 分组 + 递归切分 + overlap）
│   ├── retriever.py                        Hybrid (jieba+BM25 + dense + RRF) + 证据回链
│   ├── qa_engine.py                        三类问答（结构化引用 + 多轮改写 + Q3 conflicts）
│   ├── review_engine.py                    10 维度合同审查
│   └── llm_client.py                       Bedrock / Anthropic 后端切换
├── data/
│   └── AI知识库-综合测试文档.pdf            测试 PDF（52 页扫描件）
├── outputs/                                运行产物（gitignore）
│   ├── parsed_document.json
│   ├── chunks.json
│   ├── qa_results.json                     ← 必交输出 1
│   ├── review_results.json                 ← 必交输出 2
│   ├── chroma_db/
│   └── .ocr_cache/                         单页 OCR 缓存，重跑零成本
├── evals/
│   ├── golden.jsonl                        手工标注 10 条 QA
│   └── eval_report.json                    自动评估详细数据
├── docs/
│   ├── chunking_and_retrieval.md           分块与检索策略说明
│   ├── bad_cases.md                        失败案例与改进
│   └── evaluation.md                       自动评估报告
└── .github/workflows/ci.yml                CI（编译 + smoke test）
```

---

## 🖥️ Web Demo 功能

`streamlit run src/app.py` 后浏览器打开 [localhost:8501](http://localhost:8501)：

- **顶部状态栏**：PDF 页数 / block 数 / chunk 数 / OCR 不确定项数
- **Tab 1 问答**
  - Q1 / Q2 / Q3 三类预设问答结果（含引用展开）
  - 自由提问（简单事实 / 复杂推理两种模式，现场调 LLM）
- **Tab 2 风险审查**
  - 10 维度共 30+ 条风险，按严重度（high/medium/low）/ 类型过滤
  - 每条风险展开看支撑证据（chunk_id + 原文 quote + 章节 + 页码）
- **Tab 3 文档浏览**
  - 全部 chunks 按章节/类型/关键词过滤浏览，便于核对原文

---

## 🧠 核心流程

```
PDF (52 页扫描件)
  │
  ├─ Step 1  PyMuPDF 渲染 300 DPI → Claude Vision OCR
  │   ├─ 结构化标记 [TITLE]/[SECTION]/[TABLE_*]/[FIGURE]/[SIGNATURE]/[?]
  │   ├─ 跨页表格自动合并（表头继承）
  │   └─ 单页缓存 → outputs/.ocr_cache/
  │
  ├─ Step 2  分块
  │   ├─ table / figure / signature 独立成块（保留结构）
  │   ├─ 同 section_path 段落合并到 ≤1000 字
  │   └─ 长段按句号边界递归切分 + 120 字 overlap
  │
  ├─ Step 3  索引
  │   ├─ Dense  : ChromaDB + 本地 / OpenAI Embedding
  │   └─ Sparse : BM25 + jieba + 2-gram 兜底
  │
  ├─ Step 4  RAG 问答
  │   ├─ Q1 simple      : hybrid (RRF) + LLM rerank → 结构化 JSON 答案
  │   ├─ Q2 multi_turn  : 第 2 轮起 LLM 改写指代 → 重新检索；维护历史
  │   └─ Q3 complex     : 子问题分解 → 多路检索合并 → 输出 conflicts[]
  │                        每条标 fact / inference / human_review
  │
  └─ Step 5  合同审查（10 维度定向）
     主体一致性 · 金额一致性 · 付款 vs 验收/交付 · 交付计划一致性 · 验收标准明确性
     附件完整性 · 违约责任对等性 · 数据安全/私有化部署 · 流程图与正文一致性
     OCR 不确定项（needs_review chunks）
     ↓
     每维度独立检索 + 独立 LLM 调用 → 证据回链 → 去重 → 结构化输出
```

详细设计决策见 [`docs/chunking_and_retrieval.md`](docs/chunking_and_retrieval.md)。

---

## 📊 自动评估

```bash
python3 src/eval.py             # 跑全部（含 LLM 调用）
python3 src/eval.py --no-llm    # 仅算 Retrieval Recall
```

10 条手工标注 QA（`evals/golden.jsonl`）覆盖事实查询、列表枚举、推理判断。指标包括：

| 指标 | 含义 |
|---|---|
| Retrieval Recall@K | top-K 检索结果是否包含预期章节 |
| Citation Hit Rate | LLM 输出的 citations 是否包含预期章节 |
| 关键词覆盖率 | 答案中预期关键词的出现比例 |
| 引用回链成功率 | LLM 给的 chunk_id 直接命中真实 chunk 的比例 |

完整报告：[`docs/evaluation.md`](docs/evaluation.md)

---

## ⚙️ 环境变量

主要项（完整模板见 `.env.example`）：

```bash
# LLM 后端（二选一）
USE_BEDROCK=true                                 # AWS Bedrock：用机器 IAM 凭证
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
BEDROCK_REGION=us-east-1

# 或
USE_BEDROCK=false
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# Embedding（默认本地）
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# 检索参数
VECTOR_TOP_K=10
BM25_TOP_K=10
RERANK_TOP_K=6
```

---

## ⚠️ 限制与已知问题

1. **扫描件 PDF 必须有 Vision API**，无原生文本层；`outputs/.ocr_cache/` 让重跑零成本
2. **Vision 单页 max_tokens=8192**：极少数信息密度高的页面可能仍被截断 → 标记 needs_review
3. **印章/水印噪声**：prompt 已要求忽略；遮挡的字段用 `[?]` 占位自动进入人工复核
4. **跨页表格**：表头一致才合并；不一致时退化为独立表格 chunk
5. **本地 embedding 中文表达力有限**：BM25 + 2-gram 是主力召回；可切到 OpenAI / `bge-large-zh-v1.5`
6. **LLM JSON 输出**：内置 `_escape_inner_quotes` 处理嵌套引号、`_repair_truncated_json` 处理截断

---

## 🤝 复现说明

- 没有 Anthropic / Bedrock：可仅查 `outputs/parsed_document.json`、`outputs/chunks.json`、`outputs/qa_results.json`、`outputs/review_results.json` 验证产物
- 没有 OpenAI key：默认就是本地 embedding，不受影响
- 网络受限装不了 sentence-transformers：retriever 自动降级为纯 BM25

---

## 📜 License

仅供学习与笔试评估使用。
