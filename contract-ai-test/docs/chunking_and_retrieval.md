# Contract AI Test — 文档分块与检索策略

## 1. 文档解析方式

### PDF 扫描件处理
- **工具**: PyMuPDF (fitz) 将 PDF 每页渲染为 300 DPI 的 PNG 图像
- **OCR**: 使用 Claude Vision API (claude-sonnet-4-6) 逐页提取结构化文本
- **结构化标记**: Vision API 输出使用特殊标记区分内容类型:
  - `[TITLE]` — 章标题
  - `[SECTION]` — 节标题
  - `[TABLE_START]/[TABLE_END]` — 表格区域
  - `[LIST]` — 列表
  - `[FIGURE]` — 图示/流程图描述
  - `[SIGNATURE]` — 签署区域
  - `[PAGE: N]` — 页码标记

### 失败边界
- 若页面完全模糊/遮挡 → 标记为 `[UNREADABLE]`，置信度 0.0，进入人工复核
- 表格跨页 → 通过 page_hint 和 table_id 关联
- 扫描质量差 → confidence < 0.5 时标记需要人工复核

### 人工复核策略
- 解析后自动检查: 统计各 block_type 数量，检查标题层级连续性
- 低置信度 block 标记 `needs_review: true`
- 表格完整性校验: 对比表头行数和数据行行数是否一致

## 2. 分块策略

采用**混合分块策略**:

| 策略 | 适用内容 | 参数 |
|---|---|---|
| 标题层级切分 | 按章节组织 chunk | section_path 作为 metadata |
| 段落合并 | 同节内连续段落合并 | 最大 1200 字符 |
| 表格独立 | 表格单独成 chunk | 携带 table_id |
| 递归语义切分 | 超长段落按句子边界切 | 重叠 150 字符 |
| 图示文本化 | 流程图转文字描述 | block_type: figure_description |

### 参数
- `MAX_CHUNK_CHARS`: 1200 字符
- `OVERLAP_CHARS`: 150 字符（相邻 chunk 间重叠）

## 3. Chunk Metadata

每个 chunk 包含以下 metadata 字段:

| 字段 | 类型 | 说明 | 必填 |
|---|---|---|---|
| `section_path` | string | 章节路径，如 "第三章 > 3.2 系统功能范围" | 是 |
| `block_type` | string | 内容类型: section_title/paragraph/table/list/figure_description/signature | 是 |
| `page_hint` | int | 页码 | 是 |
| `table_id` | string | 表格唯一标识，如 "table_p3_b2" | 仅表格 |
| `source_text` | string | chunk 完整原文 | 是 |
| `table_rows` | int | 表格行数 | 仅表格 |
| `table_partial` | bool | 标记表格是否被切分 | 仅表格 |
| `chunk_index` | int | 全局 chunk 序号 | 是 |

## 4. 检索策略

### Dense Retrieval (向量检索)
- **Embedding 模型**: OpenAI `text-embedding-3-small`（默认）或本地 `all-MiniLM-L6-v2`
- **向量数据库**: ChromaDB（持久化存储）
- **相似度**: Cosine Similarity
- **top_k**: 8

### Sparse Retrieval (关键词检索)
- **算法**: BM25 (rank_bm25)
- **分词**: 简易中文分词（字+标点+2-gram）
- **top_k**: 8

### Hybrid Search (混合检索)
- **融合方式**: 加权求和 (vector_weight=0.5, bm25_weight=0.5)
- **分数归一化**: vector 距离转 similarity, BM25 分数归一化

### Rerank (重排序)
- **方法**: LLM-based rerank (Claude)
- **策略**: 对 hybrid 结果 top_k=8 用 LLM 排序，取 top_k=5
- **Prompt**: 比较各片段与查询的相关性，输出排序编号

### 过滤条件
- 支持按 `section_path` 过滤
- 支持按 `block_type` 过滤

## 5. 多轮问答策略

### 问题改写
- 第 2 轮起，对追问进行问题改写
- 使用 LLM 进行指代消解（"这些"→具体对象）、主语补全
- 改写后的 query 重新检索

### 上下文维护
- 维护完整对话历史（user + assistant）
- 回答时包含历史上下文和新增检索结果
- 引用延续: 后续轮次可引用前轮检索到的文档片段

### 策略参数
- 历史保留: 最近 4 轮对话
- 问题改写: 每轮追问前执行

## 6. 长文档策略（扩展到 200 页）

如果文档扩展到 200 页，当前策略的改进方向:

1. **分段索引**: 将文档按章节切分为多个子索引，先在章节级别路由，再在子索引内检索
2. **摘要层**: 为每个章节生成摘要，先检索摘要，锁定相关章节后再细粒度检索
3. **层次化检索**: 三层检索 — 章节级 → 段落级 → 句子级
4. **查询规划**: 复杂问题先做查询规划，分解为子查询，每个子查询路由到相关章节
5. **全局一致性校验**: 对跨章节回答做一致性检查，防止前后矛盾
6. **分页式加载**: 对超长回答分页加载，每页附带引用

## 7. 表格和图示策略

### PDF 表格
- Vision API 提取时使用 `[TABLE_START]/[TABLE_END]` 标记
- 表格独立成 chunk，携带 `table_id` 和 `table_rows`
- 跨页表格: 通过相邻页面的 table_id 合并

### 流程图和架构图
- Vision API 提取时使用 `[FIGURE]` 标记
- 输出图的文字描述（节点、箭头、关系）
- 流程图描述纳入检索范围

### 图示检索
- 图示描述文本参与 embedding 和 BM25
- 回答中引用图示时标注页码和图号

## 8. 失败边界

### 低置信度触发条件
- 页面 OCR 置信度 < 0.5
- 表格解析后行列数不一致
- 检索结果相似度 < 0.3
- LLM 回答包含"依据不足"、"无法确定"、"可能"

### 人工复核触发条件
- 合同审查中 severity=high 的风险项
- 金额相关的不一致（自动标记 `needs_human_review: true`）
- 签名页/印章页识别失败
- 跨页表格完整性无法验证

### 降级策略
- 若 embedding API 不可用 → 回退到纯 BM25
- 若 LLM API 不可用 → 输出检索片段，不生成回答
- 若 ChromaDB 不可用 → 内存模式
