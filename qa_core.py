# -*- coding: utf-8 -*-
"""
问答核心（RAG 推理链）
======================
把"检索 → 组装材料 → 生成回答 → 返回来源"这条链路集中在一处，
供命令行版（rag.py）和接口版（api.py）共用，避免同一逻辑写两遍。

对外只需：
    engine = QAEngine(collection)
    engine.answer("问题")   ->  {"question", "answer", "sources"}

检索部分用混合检索（向量 + BM25，RRF 融合），已通过评测验证
命中率优于纯向量检索（见 README 的评测章节）。
"""

import re

import requests

from hybrid_retriever import HybridRetriever

# ---------- 配置 ----------
CHAT_MODEL = "qwen2.5:3b"
MAX_DISTANCE = 0.5        # 相似度阈值（向量路粗过滤）
TOP_K = 3                 # 最终返回的片段数
NO_INFO_ANSWER = "知识库中没有找到相关信息。"

# 判"模型是不是在拒答"用的**完整话术**（提示词里已强制模型使用这个固定说法）。
# 踩坑：早期用 ("没有找到", "没有相关", ...) 这种短词表匹配，结果把
#      "dict.get() 可以避免'没有找到键'的报错" 这类正常回答也误判成拒答。
#      —— 片段越短越容易误伤，所以这里只匹配完整话术。
NO_INFO_PHRASES = (
    "知识库中没有找到相关信息",
    "知识库中没有相关信息",
    "没有找到相关信息",
)

PROMPT_TEMPLATE = (
    "你是一个知识库问答助手。请只基于下面的材料回答问题，"
    "如果材料中没有相关信息，就说'知识库中没有找到相关信息'。\n"
    "材料：\n{context}\n问题：{question}"
)


def ask(prompt, temperature=0.0):
    """调用本地大模型"""
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    })
    return resp.json()["message"]["content"]


def build_prompt(context, question):
    """把检索到的材料 + 问题拼成提示词（防幻觉约束）"""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _normalize(text):
    """去掉空白和标点，只保留文字/字母/数字，避免标点差异影响匹配"""
    return re.sub(r"[^\w]+", "", text)


def is_no_info_answer(answer):
    """判断回答是否在表示"材料里没有答案"（归一化后匹配完整拒答话术）"""
    norm = _normalize(answer)
    return any(phrase in norm for phrase in NO_INFO_PHRASES)


class QAEngine:
    """RAG 问答引擎：混合检索 + 生成回答"""

    def __init__(self, collection, max_distance=MAX_DISTANCE, top_k=TOP_K):
        self.retriever = HybridRetriever(collection, max_distance=max_distance)
        self.top_k = top_k

    def answer(self, question):
        """返回 {"question": ..., "answer": ..., "sources": [...]}"""
        hits = self.retriever.search(question, k=self.top_k)

        # 检索为空（两路都没强信号）→ 直接拒答，不调用模型
        if not hits:
            return {"question": question, "answer": NO_INFO_ANSWER, "sources": []}

        context = "\n".join(f"[来自{src}]\n{chunk}" for chunk, src, _ in hits)
        answer = ask(build_prompt(context, question))
        
        # 模型基于材料判断"材料里没有"时，来源要同步清空，避免输出自相矛盾
        # （回答"知识库中没有找到"却列着两个来源，用户会误解）
        if is_no_info_answer(answer):
            return {"question": question, "answer": answer, "sources": []}
        sources = list(dict.fromkeys(src for _, src, _ in hits))   # 保序去重

        return {"question": question, "answer": answer, "sources": sources}
