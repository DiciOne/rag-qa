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
# 说明：这里用 COPY . . 而不是逐个 COPY 文件，避免"新增模块忘记加进来"的问题
#      （曾经漏拷 qa_core.py / hybrid_retriever.py，导致容器启动即崩溃）
#      不需要的文件由 .dockerignore 排除
COPY . .

# 声明服务端口
EXPOSE 8000

# 启动命令（--host 必须是 0.0.0.0，否则容器外访问不到）
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
