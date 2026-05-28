@echo off
REM 合同 AI 审查与知识库检索 — Windows 运行脚本

cd /d "%~dp0"
echo ==============================================
echo   合同 AI 审查与知识库检索
echo ==============================================

if "%PYTHON_EXE%"=="" set PYTHON_EXE=python

echo Python  : %PYTHON_EXE%
echo Workdir : %CD%
echo.

if not exist .env (
  echo [警告] 未找到 .env，将复制 .env.example 作为模板，请编辑后再运行。
  copy .env.example .env >nul
  exit /b 1
)

if "%~1"=="" (
  %PYTHON_EXE% src\main.py --pdf "data\AI知识库-综合测试文档.pdf" --output-dir "outputs"
) else (
  %PYTHON_EXE% src\main.py %*
)
