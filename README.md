# 合同 AI 审查与知识库检索

基于"智能印章与合同审查平台建设项目"综合测试 PDF（52 页扫描件）的 RAG 问答与合同审查系统。

## 1. 能力概览

- **RAG 问答**：扫描件 PDF → 结构化文本 → 分块索引 → 三类问答（简单/多轮/复杂推理），全部回链到原文 chunk
- **合同审查**：10 个审查维度分别定向检索 + 独立 LLM 调用，证据回链到真实 `chunk_id`，severity ∈ {low, medium, high}
- **可复现**：单页 OCR 缓存 + 中间产物（parsed_document.json / chunks.json / chroma_db）全部落盘
- **降级路径**：本地 Embedding（默认）/ ChromaDB 内存模式 / 解析失败标记 needs_review

## 2. 目录结构

```
.
├── README.md                  本文件
├── .env.example               环境变量模板（.env 已 gitignore）
├── requirements.txt
├── run.sh / run.bat           Linux-macOS / Windows 启动脚本
├── src/
│   ├── main.py                主流程
│   ├── pdf_parser.py          扫描件解析（Vision OCR + 缓存 + 跨页表格合并）
│   ├── chunker.py             分块（block 分组 + 递归切分 + overlap）
│   ├── retriever.py           检索（jieba+2gram BM25 + Chroma dense + RRF + locate_evidence）
│   ├── qa_engine.py           QA（结构化引用 + 多轮改写 + Q3 conflicts 输出）
│   └── review_engine.py       审查（10 维度定向检索 + 证据回链）
├── data/
│   └── AI知识库-综合测试文档.pdf
├── outputs/                   运行产物（gitignore）
│   ├── parsed_document.json
│   ├── chunks.json
│   ├── full_text.txt
│   ├── qa_results.json        ← 必交
│   ├── review_results.json    ← 必交
│   ├── chroma_db/
│   └── .ocr_cache/            单页 OCR 缓存，重跑零成本
└── docs/
    ├── chunking_and_retrieval.md   ← 必交
    └── bad_cases.md                ← 必交
```

## 3. 环境要求

- Python 3.9+
- LLM 后端二选一：
  - **AWS Bedrock**（推荐 — 用 IAM 凭证，无需 API key），region 需开启 `claude-sonnet-4-6` 或同类 inference profile
  - **Anthropic 官方 API**（设置 `ANTHROPIC_API_KEY`）
- OpenAI API Key（**可选** — 默认使用本地 SentenceTransformer 做 embedding）

## 4. 快速开始

```bash
# 1. 安装依赖
python3 -m pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# Bedrock 用户：保持 USE_BEDROCK=true，根据机器实际 region 调整 BEDROCK_REGION
# Anthropic 直连用户：把 USE_BEDROCK 改为 false，并填 ANTHROPIC_API_KEY

# 3. 把 PDF 放到 data/ 目录（题目自带，已就位）

# 4. 一键运行
./run.sh                     # Linux/macOS
run.bat                      # Windows
```

首次运行会逐页调用 Claude Vision OCR（52 页约 3~5 分钟，~50K-150K tokens）。
所有单页 OCR 结果会缓存到 `outputs/.ocr_cache/`，**重跑零成本**。

跳过 OCR 直接用上次解析结果：
```bash
python3 src/main.py --pdf data/AI知识库-综合测试文档.pdf --skip-ocr
```

只跑 QA 不跑审查（或反之）：
```bash
python3 src/main.py --skip-ocr --no-review
python3 src/main.py --skip-ocr --no-qa
```

调试用，仅处理前 3 页：
```bash
python3 src/main.py --max-pages 3
```

## 5. 主要流程

```
PDF (52 页扫描件)
  │
  ├─ Step 1 PyMuPDF 渲染 300 DPI → Claude Vision OCR
  │  ├─ 标记标题/段落/表格/图示/签署区
  │  ├─ 跨页表格自动合并（表头继承）
  │  └─ 单页缓存 → outputs/.ocr_cache/
  │
  ├─ Step 2 分块
  │  ├─ 按 block_type 分组（table/figure/signature 独立成块）
  │  ├─ 同 section_path 段落合并到 ≤1000 字符
  │  └─ 长段落按句号边界递归切分 + 120 字 overlap
  │
  ├─ Step 3 索引
  │  ├─ Dense  : ChromaDB + 本地 SentenceTransformer / OpenAI
  │  └─ Sparse : BM25 + jieba 分词 + 中文 2-gram 兜底
  │
  ├─ Step 4 RAG 问答
  │  ├─ Q1 simple      : hybrid (RRF) + LLM rerank → 结构化 JSON 答案
  │  ├─ Q2 multi_turn  : 第 2 轮起 LLM 改写指代 → 重新检索；维护历史
  │  └─ Q3 complex     : 子问题分解 → 多路检索合并 → 输出 conflicts[]
  │                       每条标 fact / inference / human_review
  │
  └─ Step 5 合同审查（10 维度定向）
     1. 主体一致性          6. 附件完整性
     2. 金额一致性          7. 违约责任对等性
     3. 付款 vs 验收/交付   8. 数据安全/私有化部署
     4. 交付计划一致性      9. 流程图与正文一致性
     5. 验收标准明确性     10. OCR 不确定项（needs_review chunks）
     ↓
     每维度独立检索 + 独立 LLM 调用 → 证据用 retriever.locate_evidence 回链 → 去重
```

## 6. 检索策略要点

- **Hybrid Search (RRF)**：dense rank + sparse rank 用 `1/(60 + rank)` 累加，无需手动调权
- **中文分词**：`jieba.cut_for_search` + 2-gram 字符级兜底，覆盖法律术语未登录词
- **证据回链**：LLM 输出引用时给出 `chunk_id` + `quote`；后处理用 `chunk_id` 取真实元数据，`chunk_id` 错误则用 `quote` 子串/BM25 兜底反查
- **多轮改写**：保留前轮回答中的关键实体名称（系统模块名、付款节点编号等），不丢失检索锚点

详见 [`docs/chunking_and_retrieval.md`](docs/chunking_and_retrieval.md)。

## 7. 输出文件契约

### `outputs/qa_results.json`
- Q1 一条记录：`{question_id:"Q1", question_type:"simple", answer, citations[], retrieval_notes, confidence}`
- Q2 三条：`question_id:"Q2-1/2/3"`，含 `rewritten_question`、`turn`
- Q3 一条：额外字段 `conflicts[]`，每条 conflict 含 `conclusion_class ∈ {fact, inference, human_review}`

### `outputs/review_results.json`
- 数组，每条：`{risk_id, risk_type, severity, title, evidence[], reason, suggestion, needs_human_review, confidence}`
- `evidence[i]` 含 `source_id` (真实 chunk_id) / `section` / `page_hint` / `pages` / `block_type` / `table_id` / `quote`
- `severity == "high"` 自动设 `needs_human_review = true`

## 8. 限制与已知问题

1. **API 依赖**：扫描件无法本地 OCR（题目要求"扫描件材料"），首跑必须有 Anthropic API。已用 per-page 缓存把"重跑成本"压到零
2. **Vision 单页 token 上限**：max_tokens=8192；极少数信息密度高的页面可能仍被截断 → 标记需人工复核
3. **印章/水印**：Vision prompt 已要求忽略水印；印章覆盖区域用 `[?]` 占位，自动进入 `needs_review`
4. **跨页表格**：相同 section_path + 相邻页 + 表头一致才合并；不一致时退化为多个独立表格 chunk
5. **本地 embedding 中文表达力**：默认 `all-MiniLM-L6-v2` 对中文法律术语支持有限，BM25 + 2-gram 是主力召回手段
6. **LLM 输出格式**：尽量要求结构化 JSON；解析失败兜底走 quote 反查 → 几乎不会出现"无引用"的情况

## 9. 无法复现时的应对

- **没有 Anthropic key**：无法做 OCR / QA / 审查；可仅检查 `outputs/parsed_document.json` 与 `outputs/chunks.json`（如已附带）
- **没有 OpenAI key**：默认就是本地 embedding，不受影响
- **网络受限装不了 sentence-transformers**：retriever 会自动降级为纯 BM25，仍可工作

