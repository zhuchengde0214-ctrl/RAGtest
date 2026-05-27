@echo off
REM 合同 AI 审查与知识库检索 — Windows 运行脚本
REM 请先编辑 .env 文件，填入 ANTHROPIC_API_KEY

cd /d "%~dp0"
echo ========================================
echo 合同 AI 审查与知识库检索
echo ========================================
echo.
echo 确认 .env 中已配置 ANTHROPIC_API_KEY...
echo.

REM 使用项目自带的 portable Python
set PYTHON_EXE=%~dp0python-portable\python.exe

echo Python: %PYTHON_EXE%
echo.

%PYTHON_EXE% src\main.py --pdf "data\AI知识库-综合测试文档.pdf" --output-dir "outputs"

echo.
echo ========================================
echo 运行完成！
echo 结果文件在 outputs\ 目录中
echo ========================================
pause
