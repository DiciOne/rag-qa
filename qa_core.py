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

import logging
import re
import time

import requests

import config
from hybrid_retriever import HybridRetriever

# ---------- 配置 ----------
# 全部集中在 config.py，支持 .env 覆盖（模型名 / 阈值 / 超时 / 路径）。
# 这里保留同名别名，是为了让原来 `from qa_core import OLLAMA_URL` 的地方不用改。
CHAT_MODEL = config.CHAT_MODEL
OLLAMA_URL = config.OLLAMA_URL
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
MAX_DISTANCE = config.MAX_DISTANCE
TOP_K = config.TOP_K
NO_INFO_ANSWER = "知识库中没有找到相关信息。"

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """模型服务不可用（连接失败 / 超时 / 返回错误状态 / 返回格式异常）"""


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
    """调用本地大模型。失败时抛 ModelServiceError（附带可读原因）。"""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        logger.warning("模型服务响应超时（>%ss）：%s", REQUEST_TIMEOUT, OLLAMA_URL)
        raise ModelServiceError(f"模型服务响应超时（超过 {REQUEST_TIMEOUT} 秒）")
    except requests.exceptions.ConnectionError:
        logger.warning("无法连接模型服务：%s", OLLAMA_URL)
        raise ModelServiceError("无法连接模型服务，请确认 Ollama 已启动")

    # 状态码检查：不做的话，出错时会抛一个看不懂的 JSON 解析异常
    if resp.status_code != 200:
        logger.warning("模型服务返回错误状态 %s：%s", resp.status_code, resp.text[:200])
        raise ModelServiceError(
            f"模型服务返回错误状态 {resp.status_code}：{resp.text[:200]}")

    try:
        return resp.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        raise ModelServiceError(f"模型返回格式异常：{e}")


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
    """RAG 问答引擎：混合检索 + 生成回答

    store 是可选的 DocStore：传进来就会把每次问答记到 SQLite 的 query_log 表
    （提问 / 耗时 / 命中来源 / 是否拒答）。这份历史是"收集 bad case →
    改进检索 → 用评测集验证"这个闭环的入口，出问题能回放，而不是靠回忆。
    """

    def __init__(self, collection, max_distance=MAX_DISTANCE, top_k=TOP_K, store=None):
        self.retriever = HybridRetriever(collection, max_distance=max_distance)
        self.top_k = top_k
        self.store = store

    def answer(self, question):
        """返回 {"question": ..., "answer": ..., "sources": [...]}"""
        started = time.perf_counter()

        hits = self.retriever.search(question, k=self.top_k)
        retrieval_ms = int((time.perf_counter() - started) * 1000)

        # 检索为空（两路都没强信号）→ 直接拒答，不调用模型
        if not hits:
            result = {"question": question, "answer": NO_INFO_ANSWER, "sources": []}
            self._record(question, result, 0, retrieval_ms, started, refused=True)
            return result

        context = "\n".join(f"[来自{src}]\n{chunk}" for chunk, src, _ in hits)
        answer = ask(build_prompt(context, question))

        # 模型基于材料判断"材料里没有"时，来源要同步清空，避免输出自相矛盾
        # （回答"知识库中没有找到"却列着两个来源，用户会误解）
        if is_no_info_answer(answer):
            result = {"question": question, "answer": answer, "sources": []}
            self._record(question, result, len(hits), retrieval_ms, started, refused=True)
            return result

        sources = list(dict.fromkeys(src for _, src, _ in hits))   # 保序去重
        result = {"question": question, "answer": answer, "sources": sources}
        self._record(question, result, len(hits), retrieval_ms, started, refused=False)
        return result

    # ---------------- 内部：记录一条问答历史 ----------------
    def _record(self, question, result, hit_count, retrieval_ms, started, refused):
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "问答完成 | 检索=%dms 总计=%dms 命中=%d 拒答=%s | %s",
            retrieval_ms, total_ms, hit_count, refused, question[:40],
        )
        if self.store is None:
            return
        try:
            self.store.log_query(
                question=question,
                answer=result["answer"],
                sources=result["sources"],
                hit_count=hit_count,
                retrieval_ms=retrieval_ms,
                total_ms=total_ms,
                refused=refused,
            )
        except Exception as exc:
            # 记历史失败绝不能影响正常回答（日志是旁路，不是主链路）
            logger.warning("写入问答历史失败：%s", exc)
