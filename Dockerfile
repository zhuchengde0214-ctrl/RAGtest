# 合同 RAG 问答与审查 — Docker 镜像
# 用法：
#   docker build -t contract-rag .
#   docker run --rm -p 8501:8501 -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data \
#              -v $(pwd)/outputs:/app/outputs contract-rag

FROM python:3.11-slim

# 系统依赖（PyMuPDF 需要 libgl，sentence-transformers 需要 g++）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷依赖文件，利用 docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

# 拷源码
COPY src/ /app/src/
COPY evals/ /app/evals/
COPY docs/ /app/docs/
COPY README.md .env.example ./

# 默认端口
EXPOSE 8501

# 默认入口：起 Streamlit；想跑 CLI 直接覆盖 CMD
# 例：docker run ... contract-rag python3 src/main.py
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
