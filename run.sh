#!/usr/bin/env bash
# 合同 AI 审查与知识库检索 — Linux/macOS 运行脚本

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE="${PYTHON_EXE:-python3}"

echo "=============================================="
echo "  合同 AI 审查与知识库检索"
echo "=============================================="
echo "Python  : $($PYTHON_EXE --version)"
echo "Workdir : $SCRIPT_DIR"
echo

if [ ! -f .env ]; then
  echo "[警告] 未找到 .env，将复制 .env.example 作为模板，请编辑后再运行。"
  cp .env.example .env
  exit 1
fi

# 默认 PDF 路径，可被命令行参数覆盖
DEFAULT_PDF="data/AI知识库-综合测试文档.pdf"

if [ "$#" -gt 0 ]; then
  exec "$PYTHON_EXE" src/main.py "$@"
else
  exec "$PYTHON_EXE" src/main.py --pdf "$DEFAULT_PDF" --output-dir "outputs"
fi
