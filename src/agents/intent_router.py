"""IntentRouter：用轻量模型（Haiku）做意图分类。

设计理念：
- 用户的输入并不都需要走 multi-agent 流水线
- 闲聊 / 系统问题 / 无关问题用 Haiku 直接回答，省钱省时间
- 仅当问题与"合同问答 / 风险审查 / 跨合同对比"相关时才进入 PlannerAgent

输出（state）：
- state.intent: "contract_related" | "off_topic"
- state.intent_reasoning: 分类依据
- state.lite_reply: off_topic 时的直接回复（contract_related 时为空）

只用 1 次 Haiku 调用 + 极短 prompt → 成本极低（<0.001 USD/次）
"""

import json
import logging
import os
import re
from typing import Optional

from llm_client import get_lite_model, make_llm_client
from qa_engine import _escape_inner_quotes

from .base import BaseAgent
from .state import SharedState

logger = logging.getLogger(__name__)


CLASSIFY_PROMPT = """你是一个意图分类器。判断用户输入是否与「合同处理」相关。

合同相关的输入包括：
- 询问合同条款（金额、付款、交付、违约、验收等）
- 要求审查 / 风险识别 / 合规检查
- 对比新旧合同、版本差异
- 关于合同附件、表格、流程图的问题
- 关于本系统能为合同做什么的问题（这也算）

不相关的输入：
- 闲聊（"你好""今天天气怎么样"）
- 与合同无关的常识 / 编码 / 数学问题
- 攻击性 / 测试性输入

用户输入：{user_input}

输出 JSON（不要 ```）：
{{
  "intent": "contract_related" 或 "off_topic",
  "reasoning": "≤40 字判断依据"
}}"""


CHITCHAT_PROMPT = """你是合同审查系统的助手。用户输入了一个与合同无关的问题，请简短回答（≤80 字），并自然地提示一下你能为合同做什么。

可以做的事：
- 合同问答（任意条款细节）
- 风险审查（金额一致性、付款条件、违约责任、数据安全等 10 个维度）
- 跨合同对比（v1 vs v2 条款变化）

用户输入：{user_input}

直接给回答，不要加任何前缀或解释。"""


class IntentRouter(BaseAgent):
    name = "intent_router"

    def __init__(self):
        super().__init__()
        self.client = make_llm_client()
        self.model = get_lite_model()

    def check_preconditions(self, state):
        if not state.user_request.strip():
            # 没 request 直接当作 contract_related（默认全套审查）
            state.intent = "contract_related"
            state.intent_reasoning = "未提供请求，默认按合同处理流程"
            return "无 user_request，跳过分类"
        return None

    def _run(self, state: SharedState) -> None:
        # ---- Step 1: 分类 ----
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": CLASSIFY_PROMPT.format(user_input=state.user_request[:500]),
                }],
            )
            raw = resp.content[0].text
        except Exception as e:
            logger.warning(f"IntentRouter 分类失败：{e}，默认按 contract_related 处理")
            state.intent = "contract_related"
            state.intent_reasoning = f"分类调用失败：{e}"
            return

        intent, reasoning = self._parse(raw)
        state.intent = intent
        state.intent_reasoning = reasoning
        state.log(self.name, f"intent={intent}", reasoning=reasoning)

        # ---- Step 2: 如果 off_topic，直接生成回复，不再走后续 agent ----
        if intent == "off_topic":
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": CHITCHAT_PROMPT.format(user_input=state.user_request[:500]),
                    }],
                )
                state.lite_reply = resp.content[0].text.strip()
                state.log(self.name, "已生成轻量回复（off_topic）")
            except Exception as e:
                logger.warning(f"IntentRouter 闲聊回复失败：{e}")
                state.lite_reply = "您好。本系统主要用于合同问答 / 风险审查 / 跨合同对比，请就合同相关问题提问。"

    @staticmethod
    def _parse(raw: str) -> tuple[str, str]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        for cand in (text, _escape_inner_quotes(text)):
            try:
                data = json.loads(cand)
                intent = data.get("intent", "").strip()
                if intent in ("contract_related", "off_topic"):
                    return intent, data.get("reasoning", "")
            except Exception:
                pass

        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            for cand in (m.group(0), _escape_inner_quotes(m.group(0))):
                try:
                    data = json.loads(cand)
                    intent = data.get("intent", "").strip()
                    if intent in ("contract_related", "off_topic"):
                        return intent, data.get("reasoning", "")
                except Exception:
                    pass

        # 兜底：从原文里找关键词
        if "off_topic" in text.lower():
            return "off_topic", "解析兜底"
        return "contract_related", "解析失败默认"
