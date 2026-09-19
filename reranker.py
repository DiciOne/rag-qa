# -*- coding: utf-8 -*-
"""
LLM 重排序（Rerank）
====================
解决两个问题：
  1. 混合检索虽把命中率提到 1.0，但结果里混入无关内容（精确度下降）
  2. "知识库里没有答案的问题"无法准确拒答（出现 0.453 这种虚高距离）

两阶段检索：
  ① 粗排/召回：向量 + BM25 混合检索，取 top-N 候选（要"全"）
  ② 精排/重排：用大模型逐条判断"候选与问题的相关性"并打分（要"准"）
     - 最高分低于阈值 → 判定知识库无答案 → 返回 []（拒答）

代价：每次查询多一次模型调用（更慢）——"效果 vs 速度"的取舍。
"""

import re
import requests

CHAT_MODEL = "qwen2.5:3b"


def ask(prompt, temperature=0.0):
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    })
    return resp.json()["message"]["content"]


class LLMReranker:
    def __init__(self, retriever, min_score=1, snippet_len=200):
        self.retriever = retriever        # 粗排检索器（HybridRetriever）
        self.min_score = min_score        # 低于这个分数视为无关
        # 喂给模型的片段长度：必须能让模型看到完整候选内容，
        # 否则答案被截断在片段之外，模型只能回答"无"（踩过的坑）
        self.snippet_len = snippet_len

    def _score_candidates(self, question, candidates):
        """让模型选出"能回答问题"的候选，返回 {候选序号: 分数}"""
        lines = []
        for idx, (chunk, _source, _score) in enumerate(candidates):
            snippet = chunk.replace("\n", " ")[: self.snippet_len]
            lines.append(f"{idx}. {snippet}")

        prompt = (
            "下面有若干候选内容和一个问题。\n"
            "请判断：哪几条候选内容能够回答这个问题？\n"
            "只输出相关候选的编号（用逗号分隔），例如：0,3\n"
            "如果一条都不相关，只输出一个词：无\n\n"
            f"问题：{question}\n\n"
            "候选：\n" + "\n".join(lines) + "\n\n"
            "能回答该问题的候选编号："
        )
        reply = ask(prompt)

        # 情况1：模型明确说"无" → 全部 0 分 → 触发拒答
        if "无" in reply:
            return {i: 0 for i in range(len(candidates))}

        numbers = re.findall(r"\d+", reply)

        # 情况2：回复里没有任何编号（模型跑偏/乱说）→ 全部保留，避免误杀
        if not numbers:
            return {i: 1 for i in range(len(candidates))}

        # 情况3：正常情况 —— 被选中的得 1 分，其余 0 分
        scores = {i: 0 for i in range(len(candidates))}
        for n in numbers:
            idx = int(n)
            if 0 <= idx < len(candidates):      # 过滤越界编号
                scores[idx] = 1
        return scores

    def search(self, question, k=3, n_candidates=8):
        """两阶段检索：粗排取候选 → 模型重排 → 过滤低分 → 取 top-k"""
        candidates = self.retriever.search(question, k=n_candidates)
        if not candidates:
            return []

        scores = self._score_candidates(question, candidates)

        ranked = []
        for idx, (chunk, source, _old_score) in enumerate(candidates):
            score = scores.get(idx, 1)     # 模型漏打分时默认 1（保留，避免误杀）
            if score >= self.min_score:
                ranked.append((chunk, source, score))

        if not ranked:
            return []                      # 全部低分 → 知识库无答案 → 拒答

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:k]
