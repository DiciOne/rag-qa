# -*- coding: utf-8 -*-
"""
快速试问脚本（开发调试用）
==========================
用法：
    python try_qa.py                      # 用默认问题
    python try_qa.py "S7-1200 的工作电压是多少"    # 指定问题

作用：绕开命令行交互，直接调用 qa_core.QAEngine，方便快速验证效果。
"""

import sys

import chromadb
from chromadb.utils import embedding_functions

from qa_core import QAEngine

# 问题从命令行参数取（学了 sys.argv 正好用上）
question = sys.argv[1] if len(sys.argv) > 1 else "什么是RAG"

ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings", model_name="bge-m3")
collection = chromadb.PersistentClient(path="chroma_db").get_collection(
    "knowledge", embedding_function=ef)

result = QAEngine(collection).answer(question)

print("问题:", result["question"])
print("回答:", result["answer"])
print("来源:", result["sources"])
