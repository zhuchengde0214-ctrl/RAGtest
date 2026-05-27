"""
合同审查引擎

识别合同中的潜在风险，输出结构化审查结果。

审查维度：
- 主体一致性
- 金额一致性
- 付款条件与验收/交付匹配
- 交付计划一致性
- 验收标准明确性
- 附件完整性
- 违约责任对等性
- 数据安全与私有化部署
- 流程图与正文一致性
- OCR/解析不确定项

每个风险包含：
- risk_id, risk_type, severity (low/medium/high)
- evidence (来源引用)
- reason (为什么构成风险)
- suggestion (修改/复核建议)
- needs_human_review
- confidence
"""

import json
import logging
import os
from typing import Optional

from anthropic import Anthropic

from retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ReviewEngine:
    """合同审查引擎"""

    REVIEW_PROMPT = """你是一个专业的合同审查律师。请基于以下文档内容，识别合同中的潜在风险。

## 审查维度
1. 主体一致性：签约主体在正文、附件、签署页中是否一致
2. 金额一致性：合同金额在正文、报价表、SOW中是否一致
3. 付款条件与验收/交付匹配：付款节点是否与验收标准和交付计划匹配
4. 交付计划一致性：正文交付计划与SOW/附件是否一致
5. 验收标准明确性：验收标准是否具体可测量
6. 附件完整性：引用的附件是否都存在
7. 违约责任对等性：双方违约责任是否明显不对等
8. 数据安全与私有化部署：数据安全条款是否有冲突
9. 流程图与正文一致性：流程图中的责任在正文中是否落实
10. OCR/解析不确定项：因扫描件质量导致的无法确认内容

## 输出格式
请以 JSON 数组格式输出，每个风险一个对象：
```json
[
  {
    "risk_type": "金额一致性",
    "severity": "high",
    "title": "简短标题",
    "evidence_quotes": ["原文引用1", "原文引用2"],
    "evidence_sections": ["章节1", "章节2"],
    "reason": "为什么构成风险",
    "suggestion": "修改建议或复核建议"
  }
]
```

severity 必须是: low, medium, high
至少识别 6 条风险，鼓励识别更多，但不要编造没有证据的风险。
如果只是推断，在 reason 中说明"需人工确认"。

## 文档内容
{document_text}

请直接输出 JSON 数组，不要有其他内容。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY 未设置")
        self.client = Anthropic(api_key=self.api_key)

    def review(
        self,
        document_text: str,
        retriever: Optional[Retriever] = None,
        additional_contexts: Optional[list[str]] = None,
    ) -> list[dict]:
        """执行合同审查"""
        logger.info("开始合同审查...")

        # 构建完整的文档上下文
        context = document_text

        # 针对每个审查维度进行定向检索
        if retriever:
            dimension_queries = self._get_dimension_queries()
            dimension_results = {}
            for dim, query in dimension_queries.items():
                results = retriever.search(query, use_hybrid=True)
                if results:
                    dimension_results[dim] = results

            # 追加定向检索结果
            for dim, results in dimension_results.items():
                context += f"\n\n## 针对{dim}的补充检索\n"
                for r in results[:3]:
                    context += f"\n{r['content']}"

        # 调用 LLM 审查
        prompt = self.REVIEW_PROMPT.replace("{document_text}", context[:30000])

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_output = response.content[0].text

        # 解析 JSON 输出
        risks = self._parse_risks(raw_output)

        # 标准化格式
        standardized = []
        for i, risk in enumerate(risks):
            standardized.append({
                "risk_id": f"R{i + 1:03d}",
                "risk_type": risk.get("risk_type", "其他"),
                "severity": risk.get("severity", "medium"),
                "title": risk.get("title", risk.get("risk_type", "未命名风险")),
                "evidence": self._format_evidence(risk),
                "reason": risk.get("reason", ""),
                "suggestion": risk.get("suggestion", "建议人工复核"),
                "needs_human_review": self._needs_human_review(risk),
                "confidence": self._estimate_risk_confidence(risk),
            })

        logger.info(f"审查完成: 识别 {len(standardized)} 条风险")
        return standardized

    def _get_dimension_queries(self) -> dict:
        """获取各审查维度的检索查询"""
        return {
            "金额一致性": "合同金额 报价 总价 费用 支付",
            "付款与验收": "付款节点 验收条件 里程碑 付款比例",
            "交付计划": "交付物 交付时间 交付计划 SOW",
            "数据安全": "数据安全 私有化部署 等保 加密 第三方",
            "违约责任": "违约责任 赔偿 违约金 罚则",
            "附件完整性": "附件 附录 附表 签署页",
        }

    def _parse_risks(self, raw_output: str) -> list[dict]:
        """解析 LLM 输出的风险 JSON"""
        # 清洗输出
        text = raw_output.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            risks = json.loads(text)
            if isinstance(risks, list):
                return risks
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败，尝试修复...")

        # 尝试提取 JSON 数组
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error("无法解析审查结果 JSON")
        return [{
            "risk_type": "解析错误",
            "severity": "high",
            "title": "审查结果解析失败",
            "reason": "LLM 输出格式异常，需人工审查",
            "suggestion": "请人工执行合同审查",
        }]

    def _format_evidence(self, risk: dict) -> list[dict]:
        """格式化证据列表"""
        evidence = []
        quotes = risk.get("evidence_quotes", [])
        sections = risk.get("evidence_sections", [])

        for i, quote in enumerate(quotes):
            ev = {
                "source_id": f"review_evidence_{i}",
                "section": sections[i] if i < len(sections) else "未知章节",
                "quote": quote,
            }
            evidence.append(ev)

        return evidence

    def _needs_human_review(self, risk: dict) -> bool:
        """判断是否需要人工复核"""
        reason = risk.get("reason", "")
        severity = risk.get("severity", "medium")

        # 高严重性必须人工复核
        if severity == "high":
            return True
        # 原因中提到不确定、可能、推断等词
        uncertainty_keywords = ["可能", "不确定", "需确认", "推断", "OCR", "扫描", "模糊", "无法确认"]
        if any(kw in reason for kw in uncertainty_keywords):
            return True

        return False

    def _estimate_risk_confidence(self, risk: dict) -> float:
        """估计风险置信度"""
        evidence = risk.get("evidence_quotes", [])
        reason = risk.get("reason", "")

        if not evidence:
            return 0.3

        if len(evidence) >= 2:
            base = 0.8
        elif len(evidence) == 1:
            base = 0.6
        else:
            base = 0.3

        # 若有不确定性表述，降低置信度
        uncertainty_keywords = ["可能", "不确定", "需确认"]
        if any(kw in reason for kw in uncertainty_keywords):
            base -= 0.2

        return max(0.1, min(0.95, base))
