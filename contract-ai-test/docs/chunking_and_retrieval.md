# 文档分块与检索策略

## 1. 文档解析

### 1.1 扫描件 PDF 处理路径
- **工具**：PyMuPDF 渲染每页为 300 DPI PNG
- **解析模型**：Claude Vision API（`claude-sonnet-4-6`，max_tokens=8192）
- **双路径**：每页先尝试 `page.get_text()` 取原生文本层，长度 > 50 字符直接走文本层，否则才走 Vision OCR
  - 题目所给的 52 页 PDF 没有文本层，全部走 Vision
- **缓存**：单页结果缓存到 `outputs/.ocr_cache/p<NNN>_<hash>.txt`，重跑零成本

### 1.2 结构化标记 prompt
要求 LLM 用以下标记包裹不同内容（见 `pdf_parser.py:VISION_PROMPT`）：

| 标记 | 含义 |
|---|---|
| `[TITLE]` | 一级章标题 |
| `[SECTION]` | 节/小节标题 |
| `[LIST]` | 列表项 |
| `[TABLE_START] ... [TABLE_END]` | 表格（`\|` 分单元格） |
| `[FIGURE]` | 流程图/架构图的纯文字描述 |
| `[SIGNATURE]` | 签署区可读文字（含印章上的字） |
| `[?]` | 印章遮挡或模糊不清的字段占位 |
| `[UNREADABLE]` | 整页无法识别 |

prompt 中明确要求：
- 不省略金额、日期、百分比、章节编号、附件编号
- 忽略页眉页脚水印（"内部资料"等）
- 表格空单元格用 `-`

### 1.3 失败边界与人工复核
- 整页 `[UNREADABLE]` → 单 block，`confidence=0.0, needs_review=True`
- 字段含 `[?]` → 该 block `needs_review=True`
- `[TABLE_START]` 没配对 `[TABLE_END]` → block 标 `table_partial_unclosed=True, needs_review=True`
- 上述全部进入 `review_engine` 的 **OCR 不确定项** 维度，自动产出 `medium` 级风险

### 1.4 跨页表格合并
`pdf_parser._merge_cross_page_tables`：
- 同 `section_path` + 相邻页（page 或 page+1）+ 第一行表头字符串一致 → 合并
- 后续页若复制了表头，合并时自动去重
- 合并后 metadata 含 `table_pages: [3, 4, 5]` 与 `table_merged_from: [block_id, ...]` 便于回溯

## 2. 分块策略

混合分块，由 `chunker.py:DocumentChunker` 实现：

| 内容类型 | 策略 |
|---|---|
| `table` / `figure` / `signature` / `unreadable` | **独立成块**，保留完整结构 |
| `section_title` + 后续 `paragraph` / `list` | 同 `section_path` 下连续合并到 ≤ 1000 字符 |
| 单段超过 1000 字符 | 按句号 `[。！？!?\.\n]` 边界递归切分 |
| 切分后相邻 chunk | 末尾 120 字符作为 overlap 进入下一 chunk |
| 表格超长 | 保留表头 + 表头分隔行，逐段切分，标 `table_partial=True` |

参数（可在 `DocumentChunker` 改）：
- `MAX_CHUNK_CHARS = 1000`
- `MIN_MERGE_CHARS = 600`（小于此长度的合并 chunk 才会继续吸纳下一段）
- `OVERLAP_CHARS = 120`

**关键设计**：合并段落时**新建 block**，绝不修改原 `TextBlock`，避免对中间产物造成副作用。

## 3. Chunk Metadata

每个 chunk 携带的 metadata（见 `chunker._make_chunk`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_index` | int | 全局序号 |
| `section_path` | str | 形如 `第三章 系统功能 > 3.2 模块清单` |
| `block_type` | str | `paragraph / table / figure / signature / unreadable / mixed` |
| `page_hint` | int | 主页码 |
| `pages` | list[int] | 跨页时的所有页码 |
| `block_ids` | list[str] | 来源 `TextBlock` 的 id 列表（可回到原始解析） |
| `table_id` | str | 仅表格 chunk |
| `table_caption` | str | 仅表格 chunk |
| `table_rows` | int | 仅表格 chunk |
| `table_partial` | bool | 表格被拆为多个 chunk |
| `needs_review` | bool | 来源任一 block needs_review 则继承 |
| `char_len` | int | chunk 字符长度 |
| `source_text` | str | chunk 完整原文（即 content 副本，方便回溯） |

ChromaDB 不接受 `None` / `dict` / `list`，入库前用 `Retriever._sanitize_metadata` 把 `list` 转 `,` 分隔字符串、`None` 删除。

## 4. 检索策略

### 4.1 Sparse — BM25 + 中文分词
- `jieba.cut_for_search` 提供基础切词
- 额外做 **2-gram 字符级兜底**，覆盖未登录的法律术语（"违约金"、"等级保护"等）
- 标点用正则过滤，确保 token 干净
- 实现见 `retriever.tokenize_chinese`

### 4.2 Dense — Embedding + ChromaDB
- 默认本地 `sentence-transformers/all-MiniLM-L6-v2`（无需 OpenAI key）
- 也可切到 OpenAI `text-embedding-3-small`（设置 `USE_LOCAL_EMBEDDINGS=false`）
- ChromaDB cosine similarity，持久化到 `outputs/chroma_db/`，失败时自动 fallback EphemeralClient

### 4.3 Fusion — RRF
不用加权分（不同检索器分数尺度差异大），改用 **Reciprocal Rank Fusion**：
```
score(c) = Σ 1 / (rrf_k + rank_in_each_list)   (rrf_k=60)
```
返回结果带 `dense_rank` / `sparse_rank` 字段，便于 debug。

### 4.4 Rerank
可选 LLM rerank（默认开）：把 hybrid top N 喂给 Claude，要求输出排序编号序列。失败时退回 RRF 排序。

### 4.5 过滤
`search()` 支持 `filter_section`（子串匹配）/ `filter_block_type` 过滤，便于审查阶段定向召回。

## 5. 多轮问答策略

实现见 `retriever.rewrite_query` + `qa_engine.answer_multi_turn`：

1. **改写**：第 2 轮起，把当前 query 与最近 6 条历史一起发给 LLM；prompt 明确要求"保留前轮答案中已识别的关键实体"，避免改写后丢失检索锚点
2. **检索**：用改写后的 query 重新跑 hybrid，**不复用前轮检索结果**
3. **生成**：LLM 上下文里同时含 (a) 对话历史 (b) 本轮检索片段；要求基于历史维持指代
4. **引用延续**：citations 里的 `chunk_id` 可来自前轮，因为同一 retriever 实例的 `_chunk_by_id` 全局可查

## 6. 长文档策略（扩展到 200+ 页）

题目所给 52 页已工作良好。若扩展到 200+ 页：

1. **章节路由**：先用章节摘要做粗排，锁定 top-3 章节，再在章节内做细粒度检索
2. **摘要层**：每章节单独生成 200 字摘要，作为额外的 chunk 参与索引
3. **层次化检索**：query → 章节摘要 → 章节内段落 → 句子，逐层缩窄
4. **Query 规划**：复杂问题先 LLM 拆解为子问题，每个子问题路由到具体章节（已用于 Q3）
5. **跨章节一致性核查**：对涉及"金额/日期/比例"的字段做结构化抽取后做集合一致性校验
6. **引用预算**：超长上下文 LLM 易"Lost in the middle"，对每个子问题独立调 LLM 后再做汇总（review_engine 已经这么做）

## 7. 表格与图示策略

### 表格
- Vision prompt 要求 `|` 分隔单元格，空格留 `-`
- 跨页合并基于"表头字符串一致"
- 大表保留 `header + separator` 跟随每个分片，并标 `table_partial=True`

### 流程图 / 架构图
- Vision prompt 要求按"节点 → 节点"的箭头描述、关键说明文字写出
- 独立成 chunk，`block_type=figure`
- 流程图描述参与 BM25 + dense 检索；review_engine 的"流程图与正文一致性"维度专门用流程图 chunk 比对正文

### 签署区
- 单独 `block_type=signature`
- 含 `[?]` 自动 `needs_review=True`
- 进入"主体一致性"和"OCR 不确定项"两个审查维度

## 8. 失败边界与降级

| 触发条件 | 降级动作 |
|---|---|
| Vision API 调用失败 | 该页输出 `[UNREADABLE]` → 整页一个 needs_review block |
| 本地 embedding 模型加载失败 | dense 禁用，纯 BM25 检索 |
| OpenAI key 缺失或为占位 | 自动回退本地 embedding（默认就是） |
| ChromaDB 持久化失败 | EphemeralClient 内存模式 |
| LLM 答案 JSON 解析失败 | 抽取最外层 `{}` / `[]` 重试；仍失败则 answer 写明"模型输出无法解析"，confidence 设低 |
| LLM 给出的 chunk_id 在候选中找不到 | 用 quote 子串精确匹配 → BM25 兜底反查 |
| 风险无任何有效证据 | 该风险被丢弃（避免编造），不进入 review_results.json |
| `severity=high` 的风险 | 强制 `needs_human_review=true` |
