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

# ---------- 配置（与 rag.py 一致） ----------
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:3b"

import requests

def ask(prompt, temperature=0.3):
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    })
    return resp.json()["message"]["content"]

# ---------- 启动时加载知识库 ----------
ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBED_MODEL,
)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
print("✅ 知识库加载完成")

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
    return do_query(question)


@app.post("/ask")
def ask_post(req: AskRequest):
    """POST 方式提问：body 里传 {"question": "问题"}"""
    return do_query(req.question)


def do_query(question: str):
    """核心查询逻辑（复用 rag.py 的思路）"""
    results = collection.query(query_texts=[question], n_results=3)
    chunks = results["documents"][0]
    distances = results["distances"][0]

    # 相似度阈值：实测相关问题距离 ~0.39，无关问题 ~0.59，取 0.5 分隔
    relevant = [(chunk, src, dist) for chunk , src , dist in 
                zip(chunks , results["metadatas"][0],distances) if dist <0.5]
    
    if not relevant:
        return {"question":question , "answer":"知识库中没有找到相关信息。","sources":[]}

    context = "\n".join(
        f"[来自{src['source']}]\n{chunk}" for chunk, src,_ in relevant)
    
    prompt = (
        "你是一个知识库问答助手。请只基于下面的材料回答问题，"
        "如果材料中没有相关信息，就说'知识库中没有找到相关信息'。\n"
        "材料：\n" + context + "\n问题：" + question
    )
    answer = ask(prompt)
    source_names = list(set(src['source'] for _,src,_ in relevant))

    return {
        "question": question,
        "answer": answer,
        "sources": source_names,
    }
