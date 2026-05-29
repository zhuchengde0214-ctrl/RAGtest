"""
LLM 客户端工厂

支持两种 backend：
- Anthropic 官方 API (默认)
- AWS Bedrock (USE_BEDROCK=true)

返回的 client 都暴露相同的 .messages.create(...) 接口，调用方代码无需感知 backend。
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _is_bedrock() -> bool:
    return os.environ.get("USE_BEDROCK", "false").lower() in ("1", "true", "yes")


def get_default_model() -> str:
    """根据 backend 返回默认 model id"""
    if _is_bedrock():
        return os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def get_lite_model() -> str:
    """返回轻量模型（用于 IntentRouter / 简单分类 / 闲聊回复）。
    Haiku 价格约为 Sonnet 的 1/8，速度快 3 倍。
    """
    if _is_bedrock():
        return os.environ.get(
            "BEDROCK_LITE_MODEL_ID",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )
    return os.environ.get("ANTHROPIC_LITE_MODEL", "claude-haiku-4-5")


def make_llm_client(api_key: Optional[str] = None):
    """生成 LLM 客户端。
    - Bedrock: 使用 IAM/STS 凭证（不需要 api_key），区域取自 AWS_REGION 或 BEDROCK_REGION
    - Anthropic 官方: 使用 ANTHROPIC_API_KEY
    """
    if _is_bedrock():
        from anthropic import AnthropicBedrock
        region = (
            os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        logger.info(f"使用 AWS Bedrock 后端，region={region}")
        return AnthropicBedrock(aws_region=region)

    from anthropic import Anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY 未设置（或设置 USE_BEDROCK=true 改用 Bedrock）")
    if key.startswith("sk-ant-xxxx"):
        raise ValueError("ANTHROPIC_API_KEY 仍是占位值，请填入真实 key 或改用 Bedrock")
    return Anthropic(api_key=key)
