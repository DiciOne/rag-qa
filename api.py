# -*- coding: utf-8 -*-
"""
RAG 知识库问答系统 - FastAPI 接口版
====================================
把命令行版的 rag.py 变成网页接口服务

运行：uvicorn api:app --host 0.0.0.0 --port 8000
测试：浏览器访问 http://localhost:8000/docs  （自动生成的接口文档）
     或 GET  http://localhost:8000/ask?question=什么是RAG
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel

from qa_core import QAEngine        # 公共问答核心（混合检索 + 生成）

# ---------- 配置 ----------
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge"
EMBED_MODEL = "bge-m3"

# ---------- 启动时加载知识库 ----------
ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBED_MODEL,
)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
engine = QAEngine(collection)
print("✅ 知识库加载完成（混合检索模式）")

# ---------- FastAPI 应用 ----------
app = FastAPI(title="RAG 知识库问答系统", version="1.0.0")


# 定义请求体格式（POST 请求用）
class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG 知识库问答系统", "docs": "/docs", "示例": "/ask?question=什么是RAG"}


@app.get("/ask")
def ask_get(question: str):
    """GET 方式提问：/ask?question=问题"""
    return engine.answer(question)


@app.post("/ask")
def ask_post(req: AskRequest):
    """POST 方式提问：body 里传 {"question": "问题"}"""
    return engine.answer(req.question)
