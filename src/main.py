"""主入口（multi-agent 版）

底层走 src/agents/ 的 Agent 编排：
  parser → indexer → qa → audit
可以选择两种编排器：
  --orchestrator self     自研 Orchestrator（默认，依赖最少）
  --orchestrator langgraph LangGraph StateGraph

向后兼容：保留 --skip-ocr / --no-qa / --no-review 等原参数。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

from agents import Orchestrator, SharedState
from llm_client import get_default_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="合同 AI Multi-Agent 流水线")
    p.add_argument("--pdf", type=str,
                   default=os.environ.get("PDF_PATH", "data/AI知识库-综合测试文档.pdf"))
    p.add_argument("--pdf-v2", type=str, default=None,
                   help="第二份合同（用于跨合同对比 diff agent）")
    p.add_argument("--output-dir", type=str,
                   default=os.environ.get("OUTPUT_DIR", "outputs"))
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--skip-ocr", action="store_true",
                   help="复用 outputs/parsed_document.json")
    p.add_argument("--no-qa", action="store_true", help="跳过 QA agent")
    p.add_argument("--no-review", action="store_true", help="跳过审查 agent")
    p.add_argument("--request", type=str, default=None,
                   help="自然语言需求（启用 PlannerAgent，阶段 3）")
    p.add_argument("--orchestrator", choices=["self", "langgraph"],
                   default="self",
                   help="self=自研编排器；langgraph=用 LangGraph StateGraph")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    return p.parse_args()


def build_plan(args) -> list[str]:
    """根据 CLI 参数构造 plan。后续阶段 3 PlannerAgent 会替代这里。"""
    plan = ["parser", "indexer"]
    if not args.no_qa:
        plan.append("qa")
    if not args.no_review:
        plan.append("audit")
    return plan


def main():
    args = parse_args()

    use_bedrock = os.environ.get("USE_BEDROCK", "false").lower() in ("1", "true", "yes")
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not use_bedrock:
        if not api_key or api_key.startswith("sk-ant-xxxx"):
            logger.error("请在 .env 配置 ANTHROPIC_API_KEY，或设 USE_BEDROCK=true")
            sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / ".ocr_cache"

    state = SharedState(
        pdf_path=args.pdf,
        pdf_path_v2=args.pdf_v2,
        user_request=args.request or "",
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
    )

    plan = build_plan(args)

    logger.info("=" * 60)
    logger.info("合同 AI Multi-Agent — 启动")
    logger.info(f"  backend       : {'AWS Bedrock' if use_bedrock else 'Anthropic API'}")
    logger.info(f"  model         : {get_default_model()}")
    logger.info(f"  orchestrator  : {args.orchestrator}")
    logger.info(f"  PDF           : {args.pdf}")
    if args.pdf_v2:
        logger.info(f"  PDF v2        : {args.pdf_v2}")
    logger.info(f"  output        : {output_dir}")
    logger.info(f"  plan          : {plan}")
    logger.info("=" * 60)

    if args.orchestrator == "langgraph":
        # 阶段 2 实现；如未实现则回退 self
        try:
            from agents.langgraph_orchestrator import LangGraphOrchestrator
            orch = LangGraphOrchestrator(plan=plan)
        except ImportError:
            logger.warning("LangGraph 编排器未安装/未实现，回退到自研编排器")
            orch = Orchestrator(plan=plan)
    else:
        orch = Orchestrator(plan=plan)

    state = orch.run(state)

    # 全文 dump（便于人眼粗看）
    if state.parsed_doc is not None:
        (output_dir / "full_text.txt").write_text(
            state.parsed_doc.raw_text, encoding="utf-8"
        )

    if state.has_error():
        logger.error("流水线存在错误，详见上文日志")
        sys.exit(2)

    logger.info("=" * 60)
    logger.info("全部完成")
    if state.qa_results:
        logger.info(f"  QA            : {len(state.qa_results)} 条 → {output_dir/'qa_results.json'}")
    if state.risks:
        logger.info(f"  审查风险      : {len(state.risks)} 条 → {output_dir/'review_results.json'}")
    logger.info(f"  agent 日志    : {output_dir/'agent_trace.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
