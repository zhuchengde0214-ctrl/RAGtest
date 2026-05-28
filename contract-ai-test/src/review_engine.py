"""
合同审查引擎

设计原则：
1. 不做"全文塞 LLM"，而是按 10 个审查维度分别定向检索 + 独立调用 LLM
2. 每条风险的 evidence 必须回链到真实 chunk_id（用 retriever.locate_evidence 兜底）
3. severity 限定 low/medium/high；high 自动 needs_human_review=True
4. 数值类风险（金额、比例、日期）从 chunks 中预提取，作为额外 hint 传入 LLM

10 个维度：
  1. subject_consistency        主体一致性
  2. amount_consistency         金额一致性
  3. payment_vs_acceptance      付款条件与验收/交付匹配
  4. delivery_consistency       交付计划一致性
  5. acceptance_clarity         验收标准明确性
  6. attachment_completeness    附件完整性
  7. liability_balance          违约责任对等性
  8. data_security              数据安全 / 私有化部署 / 第三方
  9. flow_vs_text               流程图与正文一致性
 10. ocr_uncertainty            扫描/OCR 不确定项（needs_review=True 的 chunk）
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from llm_client import make_llm_client, get_default_model
from chunker import Chunk
from retriever import Retriever
from qa_engine import _repair_truncated_json, _escape_inner_quotes

logger = logging.getLogger(__name__)


REVIEW_DIMENSIONS = [
    {
        "id": "subject_consistency",
        "name": "主体一致性",
        "queries": ["甲方 乙方 签约主体 公司名称 全称", "签署 印章 盖章 法定代表人"],
        "focus": "甲乙双方在合同正文、SOW、报价表、附件、签署页中的名称是否完全一致；是否出现简称/全称混用；签署页主体与正文主体是否一致。",
    },
    {
        "id": "amount_consistency",
        "name": "金额一致性",
        "queries": [
            "合同总金额 总价 项目总金额 含税",
            "报价表 单价 数量 小计 合计",
            "硬件 软件 服务 实施 运维 费用",
        ],
        "focus": "正文写明的合同总金额，是否等于报价表合计、SOW 各项费用累加、附件费用清单合计；是否存在含税/不含税口径不一致；币种是否统一。",
    },
    {
        "id": "payment_vs_acceptance",
        "name": "付款条件与验收/交付匹配",
        "queries": [
            "付款节点 预付款 进度款 验收款 尾款 比例",
            "里程碑 验收 初验 终验 通过 条件",
            "支付时间 天数 工作日 银行",
        ],
        "focus": "每个付款节点是否对应明确的前置条件（合同生效/到货/初验/终验等）；付款比例累加是否=100%；不同章节描述同一节点的比例是否一致。",
    },
    {
        "id": "delivery_consistency",
        "name": "交付计划一致性",
        "queries": [
            "交付物 交付时间 交付计划 工期 进度",
            "SOW 阶段 里程碑 周 月",
            "实施 上线 部署 培训",
        ],
        "focus": "正文交付计划与 SOW 中的阶段时间、附件交付清单是否对应；交付物的命名/数量是否一致；时间线是否互相矛盾（如 SOW 第 8 周交付、正文要求第 6 周）。",
    },
    {
        "id": "acceptance_clarity",
        "name": "验收标准明确性",
        "queries": [
            "验收标准 验收方案 验收材料 测试用例",
            "性能 响应时间 准确率 SLA 指标",
            "验收报告 签字 文档",
        ],
        "focus": "验收标准是否量化可测；验收材料清单是否完整；模糊表述（如『达到客户满意』）是否需要补充。",
    },
    {
        "id": "attachment_completeness",
        "name": "附件完整性",
        "queries": [
            "附件 附录 附表 编号",
            "附件一 附件二 附件 1 附件 2",
            "技术方案 报价 安全 服务 SOW",
        ],
        "focus": "正文中引用的所有附件是否齐备；附件编号是否连续；附件标题与正文引用名称是否一致。",
    },
    {
        "id": "liability_balance",
        "name": "违约责任对等性",
        "queries": [
            "违约责任 违约金 赔偿 责任限制 上限",
            "免责 不可抗力 终止 解除",
            "保密 知识产权 侵权",
        ],
        "focus": "甲方违约责任与乙方违约责任是否明显不对等；赔偿上限/免责条款是否对一方明显不利；终止/解除条件是否单方设置。",
    },
    {
        "id": "data_security",
        "name": "数据安全与私有化部署",
        "queries": [
            "数据安全 等保 加密 隔离 私有化 本地部署",
            "云 公有云 第三方 接口 调用 SaaS",
            "个人信息 敏感数据 日志 审计",
        ],
        "focus": "私有化部署要求是否与第三方/SaaS 接口需求冲突；数据出境/共享条款是否合规；等保/加密要求是否明确到位。",
    },
    {
        "id": "flow_vs_text",
        "name": "流程图与正文一致性",
        "queries": [
            "流程图 流程 步骤 环节 节点",
            "审批 复核 人工 自动",
            "审查 印章 用印 签批",
        ],
        "focus": "流程图中的人工复核/审批环节是否在正文中明文落实；流程图中的关键节点（如双人复核、领导审批）是否在职责章节体现。",
    },
    {
        "id": "ocr_uncertainty",
        "name": "OCR/解析不确定项",
        "queries": [],  # 用 needs_review=True 的 chunk 直接构造
        "focus": "因扫描质量、印章遮挡、跨页表格等原因导致 OCR 不可靠的字段，必须进入人工复核。",
    },
]


PROMPT_TPL = """你是一个资深的合同审查律师。请仅基于以下与「{dim_name}」相关的文档片段，识别该维度的潜在风险。

## 审查重点
{focus}

## 已编号文档片段
{chunks}

## 输出要求
- 输出 JSON 数组（不要 ```），最多 4 条最重要的风险，每条一个对象
- 没有发现风险则输出 []
- 不要编造；如果只是推断，设置 needs_human_review=true
- severity 只能是 low / medium / high
- chunk_id 必须从片段头部 [chunk_id] 复制；quote 必须是片段中连续原文（≤80 字）
- reason 控制在 100 字内；suggestion 控制在 60 字内

## JSON 结构
[
  {{
    "risk_type": "{dim_name}",
    "severity": "low|medium|high",
    "title": "≤25 字标题",
    "evidence": [
      {{"chunk_id": "...", "quote": "...", "section": "..."}}
    ],
    "reason": "≤100 字",
    "suggestion": "≤60 字",
    "needs_human_review": true|false,
    "confidence": 0.0~1.0
  }}
]"""


class ReviewEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        per_dim_top_k: int = 8,
    ):
        self.model = model or get_default_model()
        self.per_dim_top_k = per_dim_top_k
        self.client = make_llm_client(api_key=api_key)

    # ------------------------------------------------------------------
    def review(self, chunks: list[Chunk], retriever: Retriever) -> list[dict]:
        logger.info("开始合同审查（分维度定向检索）")
        all_risks: list[dict] = []

        for dim in REVIEW_DIMENSIONS:
            logger.info(f"  [审查维度] {dim['name']}")
            if dim["id"] == "ocr_uncertainty":
                risks = self._review_ocr_uncertainty(chunks, retriever)
            else:
                relevant = self._collect_relevant(retriever, dim["queries"])
                if not relevant:
                    logger.info(f"    跳过：未检索到相关片段")
                    continue
                risks = self._review_one_dim(dim, relevant, retriever)

            for r in risks:
                r["risk_type"] = dim["name"]
            all_risks.extend(risks)

        # 去重 + 编号
        all_risks = self._dedup(all_risks)
        for i, r in enumerate(all_risks):
            r["risk_id"] = f"R{i + 1:03d}"
            # 字段顺序整理
            r.setdefault("needs_human_review", r.get("severity") == "high")
            r.setdefault("confidence", 0.6)

        logger.info(f"审查完成，共 {len(all_risks)} 条风险")
        return all_risks

    # ------------------------------------------------------------------
    def _collect_relevant(self, retriever: Retriever, queries: list[str]) -> list[dict]:
        merged: dict[str, dict] = {}
        for q in queries:
            res = retriever.search(q, top_k=self.per_dim_top_k)
            for r in res:
                cid = r["chunk_id"]
                if cid not in merged:
                    merged[cid] = {**r, "matched_queries": [q]}
                else:
                    merged[cid]["matched_queries"].append(q)
                    merged[cid]["score"] += r["score"]
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        # 控制 prompt 长度
        return ranked[: self.per_dim_top_k * 2]

    # ------------------------------------------------------------------
    def _format_chunks(self, chunks: list[dict]) -> str:
        out = []
        for c in chunks:
            m = c.get("metadata", {})
            head = (
                f"[{c['chunk_id']}] section={m.get('section_path', '') or '?'}"
                f" | type={m.get('block_type', '')}"
                f" | pages={m.get('pages', m.get('page_hint'))}"
            )
            if m.get("table_id"):
                head += f" | table_id={m['table_id']}"
            out.append(f"{head}\n{c['content']}")
        return "\n\n---\n\n".join(out)

    def _review_one_dim(self, dim: dict, candidates: list[dict], retriever: Retriever) -> list[dict]:
        prompt = PROMPT_TPL.format(
            dim_name=dim["name"],
            focus=dim["focus"],
            chunks=self._format_chunks(candidates),
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "max_tokens":
                logger.warning(f"  维度 {dim['name']} 输出达 max_tokens 上限，可能被截断；将用 JSON repair 兜底")
        except Exception as e:
            logger.error(f"维度 {dim['name']} 调用 LLM 失败: {e}")
            return []

        risks_raw = self._parse_json_array(raw)
        out = []
        for r in risks_raw:
            if not isinstance(r, dict):
                continue
            sev = r.get("severity", "medium")
            if sev not in ("low", "medium", "high"):
                sev = "medium"
            evidence = self._normalize_evidence(r.get("evidence") or [], candidates, retriever)
            if not evidence:
                # 没有有效证据的风险丢弃，避免编造
                logger.info(f"    丢弃无证据风险: {r.get('title', '')}")
                continue
            needs_human = bool(r.get("needs_human_review", sev == "high"))
            conf = r.get("confidence")
            try:
                conf = float(conf) if conf is not None else 0.7
            except Exception:
                conf = 0.7
            out.append({
                "severity": sev,
                "title": r.get("title", "")[:60],
                "evidence": evidence,
                "reason": r.get("reason", "")[:600],
                "suggestion": r.get("suggestion", "")[:300],
                "needs_human_review": needs_human or sev == "high",
                "confidence": max(0.0, min(1.0, conf)),
            })
        return out

    # ------------------------------------------------------------------
    def _review_ocr_uncertainty(self, chunks: list[Chunk], retriever: Retriever) -> list[dict]:
        flagged = [c for c in chunks if c.metadata.get("needs_review")]
        if not flagged:
            return []
        # 把同章节的 needs_review chunk 聚合成一条风险
        by_section: dict[str, list[Chunk]] = {}
        for c in flagged:
            sec = c.metadata.get("section_path", "(未知章节)")
            by_section.setdefault(sec, []).append(c)

        out = []
        for sec, group in by_section.items():
            ev = []
            for c in group[:5]:
                ev.append({
                    "source_id": c.chunk_id,
                    "section": sec,
                    "page_hint": c.metadata.get("page_hint"),
                    "pages": c.metadata.get("pages"),
                    "block_type": c.metadata.get("block_type"),
                    "table_id": c.metadata.get("table_id"),
                    "quote": c.content[:120],
                    "resolved": True,
                })
            out.append({
                "severity": "medium",
                "title": f"OCR/解析不确定项：{sec[:30]}",
                "evidence": ev,
                "reason": (
                    f"该章节有 {len(group)} 处片段被标记 needs_review=True（如印章遮挡、跨页表头缺失、"
                    f"扫描质量低或字符出现 [?] 占位），可能影响关键字段的准确性。"
                ),
                "suggestion": "对照 PDF 原件人工复核相关章节，特别确认金额、日期、主体名称等关键字段。",
                "needs_human_review": True,
                "confidence": 0.85,
            })
        return out

    # ------------------------------------------------------------------
    def _normalize_evidence(self, ev_raw: list, candidates: list[dict], retriever: Retriever) -> list[dict]:
        if not isinstance(ev_raw, list):
            return []
        cand_by_id = {c["chunk_id"]: c for c in candidates}
        out = []
        for e in ev_raw:
            if not isinstance(e, dict):
                continue
            cid = (e.get("chunk_id") or "").strip()
            quote = (e.get("quote") or "").strip()
            chunk = cand_by_id.get(cid)
            if chunk is None and quote:
                located = retriever.locate_evidence(quote, top_n=1)
                if located:
                    cid = located[0]["chunk_id"]
                    chunk = cand_by_id.get(cid)
                    if chunk is None:
                        chunk = {
                            "chunk_id": cid,
                            "content": located[0]["snippet"],
                            "metadata": {
                                "section_path": located[0]["section"],
                                "page_hint": located[0]["page_hint"],
                                "pages": located[0]["pages"],
                                "block_type": located[0]["block_type"],
                                "table_id": located[0]["table_id"],
                            },
                        }
            if chunk is None:
                # 无法回链，跳过
                continue
            m = chunk.get("metadata", {})
            out.append({
                "source_id": chunk["chunk_id"],
                "section": m.get("section_path", "") or e.get("section", ""),
                "page_hint": m.get("page_hint"),
                "pages": m.get("pages"),
                "block_type": m.get("block_type"),
                "table_id": m.get("table_id"),
                "quote": quote[:200],
                "resolved": True,
            })
        return out

    # ------------------------------------------------------------------
    def _parse_json_array(self, text: str) -> list:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        def _try(s: str):
            try:
                d = json.loads(s)
                return d if isinstance(d, list) else None
            except Exception:
                return None

        # 1) 直接
        d = _try(text)
        if d is not None:
            return d
        # 2) 抽取数组
        m = re.search(r"\[\s*[\s\S]*\]\s*$", text)
        cand = m.group(0) if m else text
        d = _try(cand)
        if d is not None:
            return d
        # 3) 转义内部裸引号
        d = _try(_escape_inner_quotes(cand))
        if d is not None:
            logger.info("  风险 JSON 修复（内部引号），解析出 %d 条", len(d))
            return d
        # 4) 截断兜底
        repaired = _repair_truncated_json(text)
        if repaired:
            d = _try(repaired)
            if d is None:
                d = _try(_escape_inner_quotes(repaired))
            if d is not None:
                logger.info("  风险 JSON 截断已修复，解析出 %d 条", len(d))
                return d
        # 把完整原始输出 dump 到文件方便诊断
        try:
            import time
            tmp = f"/tmp/review_parse_failed_{int(time.time()*1000)}.txt"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            logger.warning("无法解析风险 JSON（完整原始输出已写入 %s）；前 300 字: %r", tmp, text[:300])
        except Exception:
            logger.warning("无法解析风险 JSON，原始输出片段: %r", text[:300])
        return []

    # ------------------------------------------------------------------
    def _dedup(self, risks: list[dict]) -> list[dict]:
        """去重：title + 主要证据 chunk 相同的视为重复"""
        seen = set()
        out = []
        for r in risks:
            key_chunks = tuple(sorted(e.get("source_id", "") for e in r.get("evidence", [])))
            key = (r.get("title", "")[:30], key_chunks)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
