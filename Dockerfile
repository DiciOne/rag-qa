# RAG 知识库问答系统 - Docker 镜像构建文件
# 构建：docker build -t rag-qa .
# 运行：docker run -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db rag-qa

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 分层缓存：依赖不变时不会重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY api.py .
COPY rag.py .
COPY knowledge/ ./knowledge/

# 声明服务端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
