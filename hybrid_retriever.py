# -*- coding: utf-8 -*-
"""
混合检索（Hybrid Search）
==========================
原理：向量检索擅长"语义相近"（同义改写），关键词检索（BM25）擅长"字面精确匹配"
     （设备型号、专业术语、数字）。两路结果融合，互相补短板。

为什么需要：评测发现"数字量输入有多少点"这类含专业词的提问，纯向量检索会把
           正确内容排到第 3 位（距离 0.509 被阈值挡住），而关键词检索能直接命中。

融合算法：RRF（Reciprocal Rank Fusion，倒数排名融合）
      score(chunk) = Σ 1 / (60 + rank)，rank 从 1 开始
      —— 只关心"排第几"，不关心分数绝对值，所以两路分数不可比也能融合。
"""

import jieba
from rank_bm25 import BM25Okapi

import config


class HybridRetriever:
    def __init__(self, collection, max_distance=None):
        self.collection = collection
        # 阈值默认取 config（可在 .env 覆盖）；显式传参时以传参为准
        self.max_distance = config.MAX_DISTANCE if max_distance is None else max_distance

        # 取出库里所有块，用于构建 BM25 索引
        data = collection.get(include=["documents", "metadatas"])
        self.chunks = data["documents"]
        self.sources = [m["source"] for m in data["metadatas"]]

        # BM25 需要分词（中文必须先切词）
        self.tokenized = [list(jieba.cut(doc)) for doc in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    # ---------- 第一路：向量检索 ----------
    def vector_search(self, question, n=10):
        """返回 [(块内容, 来源, 距离), ...]，按距离升序，不做阈值过滤"""
        n = min(n, len(self.chunks))
        r = self.collection.query(query_texts=[question], n_results=n)
        out = []
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
            out.append((doc, meta["source"], dist))
        return out

    # ---------- 第二路：关键词检索（BM25）----------
    def keyword_search(self, question, n=10):
        """返回 [(块内容, 来源, BM25分数), ...]，按分数降序"""
        tokens = list(jieba.cut(question))
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.chunks[i], self.sources[i], float(scores[i])) for i in ranked]

    # ---------- 融合 ----------
    def search(self, question, k=3, n_candidates=10):
        """混合检索：向量 + BM25 两路召回 → RRF 融合 → 返回 [(块内容, 来源, 得分), ...]

        拒答规则：向量最优距离超过阈值 → 返回 []（只看距离，不看 BM25，原因见下）。
        """
        vec = self.vector_search(question, n_candidates)
        kw = self.keyword_search(question, n_candidates)

        # 两路都是"总是返回 n 条"（不做阈值过滤），
        # 所以列表为空只可能是库为空，而不是"没检索到"
        if not vec:
            return []

        # ---------- 拒答判定：只看向量最优距离 ----------
        #
        # ★ 旧规则是「向量距离超标 且 BM25 最高分 == 0」，在 2822 块的真实
        #   知识库上直接失效 —— 库外问题一条都拒不掉，拒答率从 0.67 掉到 0.00。
        #
        #   原因：库一大，几乎任何中文问题都能在某个块里命中几个常用词。实测：
        #     "怎么用 Excel 做数据透视表"  BM25 = 18.078  ← 因为命中"数据"
        #     "PROFINET 通信怎么配置"      BM25 = 11.823  ← 库内问题反而更低
        #   所以 "BM25 == 0" 这个条件永远不成立，整条规则等于没写。
        #
        #   同时向量距离分得很开（2822 块真实库，19 条问题实测 top1）：
        #     库内问题：0.239 ~ 0.433
        #     库外问题：0.453 ~ 0.558
        #   扫阈值 0.30~0.60，"总错误数（误拒+误答）"最小的是 0.44~0.45（=1）。
        #   见 config.MAX_DISTANCE 的注释。
        #
        #   结论：BM25 负责「召回与排序」，该不该拒答只由向量距离判断 ——
        #   两路各司其职，不要让 BM25 兼管"有没有信号"。
        if vec[0][2] > self.max_distance:
            return []

        scores = {}
        source_of = {}

        for rank , (chunk , source,_) in enumerate(vec,start = 1):
            scores[chunk] = scores.get(chunk , 0) + 1 / (60+rank)
            source_of[chunk] = source
        for rank , (chunk , source,_) in enumerate(kw,start=1):
            scores[chunk] = scores.get(chunk , 0) +1 / (60+rank)
            source_of[chunk] = source

        sorted_scores = sorted(scores.items() , key=lambda x:x[1] , reverse=True)
        result = []
        for chunk , score in sorted_scores[:k]:
            result.append((chunk,source_of[chunk] , score))
        return result
  