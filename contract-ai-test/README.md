# 合同 AI 审查与知识库检索

基于"智能印章与合同审查平台建设项目"综合测试文档的 RAG 问答与合同审查系统。

## 项目概述

本项目实现了两个核心能力：

1. **RAG 问答**：将 PDF 合同文档处理为可检索的知识库，支持简单事实查询、多轮对话和跨文档复杂推理。
2. **合同审查**：自动识别合同中的潜在风险，输出结构化审查结果（至少 6 条，含证据和建议）。

## 目录结构

```
contract-ai-test/
  README.md                       # 本文件
  .env.example                    # 环境变量配置示例
  requirements.txt                # Python 依赖
  src/                            # 源码
    __init__.py
    main.py                       # 主入口
    pdf_parser.py                 # PDF 解析 (OCR/视觉模型)
    chunker.py                    # 文档分块
    retriever.py                  # 检索 (向量+BM25+Rerank)
    qa_engine.py                  # RAG 问答引擎
    review_engine.py              # 合同审查引擎
  outputs/                        # 输出文件
    qa_results.json               # RAG 问答结果
    review_results.json           # 合同审查结果
  docs/                           # 文档
    chunking_and_retrieval.md     # 分块与检索策略说明
    bad_cases.md                  # 失败案例分析 (>= 3 个)
  data/                           # 数据
    AI知识库-综合测试文档.pdf      # 测试 PDF (自行放入)
```

## 环境要求

- Python 3.11+
- 以下 API Key 之一:
  - **Anthropic API Key**（必需 — 用于 OCR 文本提取、问答、审查）
  - **OpenAI API Key**（可选 — 用于 Embedding，也可使用本地模型替代）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 准备 PDF

将 `【玄武纪】综合测试文档.pdf` 放入 `data/` 目录。

### 4. 运行

```bash
cd src
python main.py --pdf ../data/AI知识库-综合测试文档.pdf --output-dir ../outputs
```

### 5. 查看结果

- `outputs/qa_results.json` — 三类问答结果
- `outputs/review_results.json` — 合同审查风险列表
- `outputs/parsed_document.json` — 解析后的结构化文档
- `outputs/chunks.json` — 分块结果

## 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--pdf` | PDF 文件路径（必需） | - |
| `--output-dir` | 输出目录 | `../outputs` |
| `--skip-ocr` | 跳过 OCR，使用已保存的解析结果 | `false` |
| `--parsed-json` | 已解析文档 JSON 路径 | - |
| `--api-key` | Anthropic API Key | 环境变量 `ANTHROPIC_API_KEY` |
| `--model` | Claude 模型名称 | `claude-sonnet-4-6` |

## 主要流程

```
PDF (扫描件)
  ↓ PyMuPDF 渲染为图像 (300 DPI)
  ↓ Claude Vision API 逐页提取结构化文本
  ↓ 解析为 TextBlock (标题/段落/表格/图示/签署)
  ↓ 分块 (按节合并段落, 表格独立, 递归切分长文本)
  ↓ 索引 (ChromaDB 向量 + BM25 关键词)
  ↓ 检索 (Hybrid + LLM Rerank)
  ↓ ├─ RAG 问答 (Q1简单/Q2多轮/Q3复杂推理)
  ↓ └─ 合同审查 (10 维度定向检索 + LLM 审查)
  ↓ 输出 JSON 结果
```

## 检索策略概要

- **Embedding**: OpenAI `text-embedding-3-small` 或本地 `all-MiniLM-L6-v2`
- **向量库**: ChromaDB (Cosine Similarity)
- **关键词**: BM25 (rank_bm25)
- **融合**: 加权求和 (vector 0.5 + BM25 0.5)
- **Rerank**: LLM-based
- **多轮**: LLM 问题改写 + 对话历史维护

详见 [`docs/chunking_and_retrieval.md`](docs/chunking_and_retrieval.md)。

## 限制与已知问题

1. **API 依赖**: PDF OCR 依赖 Claude Vision API，每页约 1-2 秒，20 页约 20-40 秒
2. **API 成本**: 每页 OCR 约消耗 ~2000-4000 tokens（图像），整个流程约 50K-100K tokens
3. **OCR 精度**: 扫描件质量直接影响提取准确率，低质量页面会进入人工复核
4. **表格处理**: 复杂嵌套表格可能提取不完整
5. **本地运行**: 如果无法访问 API，系统无法完成 OCR 和问答（文本替换为本地 OCR 方案需额外配置 Tesseract）
6. **嵌入模型**: 通用 embedding 模型对中文法律术语的语义理解有限

## 如果评审人员无法完全复现

本项目依赖以下外部服务，如果因网络或账号限制无法访问：

1. **Anthropic API**: 可替换为其他兼容 API（OpenAI、本地模型等）
2. **OpenAI API**: Embedding 可切换为本地模型（设置 `USE_LOCAL_EMBEDDINGS=true`）
3. **ChromaDB**: 会自动降级为内存模式

所有中间结果（解析后的 JSON、chunks、检索片段）都会保存到 `outputs/` 目录，评审人员可以直接查看。

## 扩展性

系统设计支持扩展到 200+ 页文档：
- 章节级路由 + 子索引
- 层次化检索（章节→段落→句子）
- 摘要层加速
- 查询规划和分解

详见 `docs/chunking_and_retrieval.md` 第 6 节。

## License

本项目仅用于笔试评估目的。
