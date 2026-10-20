# -*- coding: utf-8 -*-
"""
RAG 知识库问答系统 - FastAPI 接口版
====================================
把命令行版的 rag.py 变成网页接口服务

运行：uvicorn api:app --host 0.0.0.0 --port 8000
测试：浏览器访问 http://localhost:8000/docs  （自动生成的接口文档）
     或 GET  http://localhost:8000/ask?question=什么是RAG
"""

import logging

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from doc_store import DocStore
from qa_core import QAEngine, ModelServiceError   # 公共问答核心（混合检索 + 生成）

# ---------- 配置（全部来自 config.py，支持 .env 覆盖）----------
CHROMA_DIR = config.CHROMA_DIR
COLLECTION_NAME = config.COLLECTION_NAME
EMBED_MODEL = config.EMBED_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- 启动时加载知识库 ----------
ef = embedding_functions.OllamaEmbeddingFunction(
    url=config.EMBED_URL,
    model_name=EMBED_MODEL,
)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
engine = QAEngine(collection, store=DocStore())   # 接口请求也会写进 query_log
logger.info("知识库加载完成 | %s", config.describe())
print("✅ 知识库加载完成（混合检索模式）")

# ---------- FastAPI 应用 ----------
app = FastAPI(title="RAG 知识库问答系统", version="1.0.0")


@app.exception_handler(ModelServiceError)
def handle_model_error(request: Request, exc: ModelServiceError):
    """模型服务不可用时返回 503，而不是 500 崩溃（部署时监控系统能识别）"""
    return JSONResponse(
        status_code=503,
        content={"error": "模型服务暂不可用", "detail": str(exc)},
    )


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


@app.get("/stats")
def stats():
    """知识库统计 + 问答历史统计（总次数 / 拒答率 / 平均耗时）"""
    return {
        "knowledge_base": engine.store.stats(),
        "queries": engine.store.query_stats(),
    }


@app.get("/history")
def history(limit: int = 10):
    """最近的问答记录 —— 排查"刚才那个问题为什么答错"时直接用"""
    return {"items": engine.store.recent_queries(limit)}
