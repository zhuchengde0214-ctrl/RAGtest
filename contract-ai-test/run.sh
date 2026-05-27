#!/bin/bash
# 合同 AI 审查与知识库检索 — 运行脚本
# 请先编辑 .env 文件，填入 ANTHROPIC_API_KEY

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "合同 AI 审查与知识库检索"
echo "========================================"
echo ""
echo "确认 .env 中已配置 ANTHROPIC_API_KEY..."
echo ""

PYTHON_EXE="$SCRIPT_DIR/python-portable/python.exe"
echo "Python: $PYTHON_EXE"
echo ""

"$PYTHON_EXE" src/main.py --pdf "data/AI知识库-综合测试文档.pdf" --output-dir "outputs"

echo ""
echo "========================================"
echo "运行完成！"
echo "结果文件在 outputs/ 目录中"
echo "========================================"
