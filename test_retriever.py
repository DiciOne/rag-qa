# -*- coding: utf-8 -*-
"""
混合检索「拒答规则」回归测试
=============================
跑法：pytest test_retriever.py -v

不需要 Ollama、不需要 Chroma —— 用假的 vector_search / keyword_search
喂进构造好的距离和分数，直接验证 search() 的拒答判定。

★ 为什么要有这个测试
------------------------------------------------------------
旧规则是「向量距离超标 且 BM25 最高分 == 0」。在 18 块的小库上没问题，
换到 2822 块的真实知识库（西门子 + 台达手册）后拒答率直接归零：

  库外问题实测：
    "今天天气怎么样"            BM25 = 0        ← 唯一能拒掉的
    "公司食堂几点开门"          BM25 = 6.922
    "怎么用 Excel 做数据透视表"  BM25 = 18.078   ← 比库内问题还高
    "帮我写一首诗"              BM25 = 10.623
    "红烧肉怎么做"              BM25 = 6.608

  库内问题：
    "PROFINET 通信怎么配置"     BM25 = 11.823   ← 居然比上面的还低

库一大，几乎任何中文问题都能在某个块里命中几个常用词，所以
"BM25 == 0" 永远不会成立 —— 整条规则形同虚设。

修复：该不该拒答只看向量距离，BM25 只负责召回与排序。
本测试锁住这个决定：**只要向量距离超阈值，即使 BM25 分数很高也必须拒答。**
"""

import pytest

from hybrid_retriever import HybridRetriever


class FakeRetriever(HybridRetriever):
    """绕过 __init__（不连真的 Chroma），只测 search() 的判定逻辑"""

    def __init__(self, vec_hits, kw_hits, max_distance):
        self.max_distance = max_distance
        self._vec_hits = vec_hits
        self._kw_hits = kw_hits

    def vector_search(self, question, n=10):
        return self._vec_hits

    def keyword_search(self, question, n=10):
        return self._kw_hits


def test_refuses_even_when_bm25_score_is_high():
    """核心回归：向量距离超标时，BM25 分数再高也必须拒答

    这正是旧规则失效的场景 —— "怎么用 Excel 做数据透视表" BM25 = 18.078。
    """
    vec = [("块A", "a.pdf", 0.50)]
    kw = [("块A", "a.pdf", 18.078)]
    r = FakeRetriever(vec, kw, max_distance=0.44)
    assert r.search("怎么用 Excel 做数据透视表") == []


def test_refuses_plainly_when_bm25_is_zero():
    """BM25 为 0 且距离超标 → 当然拒答（旧规则也能过的一条）"""
    vec = [("块A", "a.pdf", 0.60)]
    kw = [("块A", "a.pdf", 0.0)]
    r = FakeRetriever(vec, kw, max_distance=0.44)
    assert r.search("今天天气怎么样") == []


def test_answers_when_distance_is_within_threshold():
    """距离在阈值内 → 正常返回结果，不能拒答"""
    vec = [("块A", "a.pdf", 0.30), ("块B", "b.pdf", 0.35)]
    kw = [("块A", "a.pdf", 12.0)]
    r = FakeRetriever(vec, kw, max_distance=0.44)
    hits = r.search("S7-1200 的工作电压是多少")
    assert len(hits) > 0
    assert any(src == "a.pdf" for _, src, _ in hits)


def test_threshold_boundary_is_strictly_greater():
    """边界：距离**恰好等于**阈值时不拒答（规则是 > 才拒）"""
    vec = [("块A", "a.pdf", 0.44)]
    kw = [("块A", "a.pdf", 5.0)]
    r = FakeRetriever(vec, kw, max_distance=0.44)
    assert r.search("边界问题") != []


def test_empty_collection_returns_empty():
    """库为空（两路都返回空）→ 返回 []，而不是报错"""
    r = FakeRetriever([], [], max_distance=0.44)
    assert r.search("任何问题") == []


@pytest.mark.parametrize(
    "distance, should_refuse",
    [
        (0.239, False),
        (0.395, False),
        (0.433, False),
        (0.441, True),
        (0.453, True),
        (0.558, True),
    ],
    ids=[
        "库内-最相关",
        "库内-列表推导式",
        "库内-最不相关的一条(messages)",
        "库外-刚过阈值",
        "库外-实测最低的一条(Excel)",
        "库外-很无关(红烧肉/Chroma)",
    ],
)
def test_distances_from_real_measurements(distance, should_refuse):
    """用真实实测的距离值验证阈值 0.44 的判定

    数值来自 2822 块真实库上用 19 条评测问题扫出来的 top1 距离
    （见 config.MAX_DISTANCE 的注释）：
      库内 0.239~0.433    库外 0.453~0.558
    """
    vec = [("块A", "a.pdf", distance)]
    kw = [("块A", "a.pdf", 18.078)]      # 故意给个很高的 BM25，验证它不影响判定
    r = FakeRetriever(vec, kw, max_distance=0.44)
    refused = r.search("测试问题") == []
    assert refused is should_refuse


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
