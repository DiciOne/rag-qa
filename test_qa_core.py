# -*- coding: utf-8 -*-
"""
问答核心回归测试
================
跑法：
    pytest test_qa_core.py -v         # 推荐
    python test_qa_core.py            # 也可以

覆盖"模型拒答判定"（qa_core.is_no_info_answer）的边界用例。
纯函数测试，不需要 Ollama，秒级跑完。每次改拒答判定或提示词都跑一遍。

背景：拒答判定踩过两个坑，这里的用例就是为它们锁的
  1. 用短词表 ("没有找到", "没有相关", ...) 匹配
     → 把含"没有找到键"的正常回答误判成拒答，来源被错误清空
  2. 直接"精确等于固定话术"
     → 模型实测会丢掉句末标点，导致漏判
"""

import pytest

from qa_core import is_no_info_answer


# ============================================================
# 用例组 1：应判为「拒答」→ 来源要清空
# ============================================================
SHOULD_REFUSE = [
    ("标准话术（带句号）", "知识库中没有找到相关信息。"),
    ("标准话术（无句号——模型实测会丢标点）", "知识库中没有找到相关信息"),
    ("标准话术 + 附带建议", "知识库中没有找到相关信息。建议补充相关文档。"),
    ("标准话术 + 空格换行干扰", "知识库中没有找到\n相关信息。"),
    ("变体：缺「知识库中」", "没有找到相关信息。"),
    ("变体：没有「找到」", "知识库中没有相关信息。"),
]


@pytest.mark.parametrize(
    "answer", [a for _, a in SHOULD_REFUSE], ids=[n for n, _ in SHOULD_REFUSE]
)
def test_detects_refusal(answer):
    """模型在拒答 → 判定为 True（外层据此清空来源）"""
    assert is_no_info_answer(answer) is True


# ============================================================
# 用例组 2：不应判为「拒答」→ 来源要保留
# 这几条是回归用例：旧的短词表会把它们全部误判成拒答
# ============================================================
SHOULD_KEEP = [
    (
        "含「没有找到」但是有效回答",
        "dict.get() 可以在键不存在时返回 None，避免'没有找到键'的报错。",
    ),
    ("含「未找到」但是有效回答", "字典取值时如果键未找到，用 get() 返回默认值。"),
    ("含「无法确定」但是有效回答", "无法确定具体型号时，可参考手册中的 S7-1200 参数表。"),
    ("普通有效回答", "RAG 代表检索增强生成（Retrieval-Augmented Generation）。"),
]


@pytest.mark.parametrize(
    "answer", [a for _, a in SHOULD_KEEP], ids=[n for n, _ in SHOULD_KEEP]
)
def test_keeps_valid_answer(answer):
    """正常回答里只是恰好出现"没有找到"这类字眼 → 判定为 False，来源必须保留"""
    assert is_no_info_answer(answer) is False


# ============================================================
# 已知残留风险：用 xfail 显式标记，而不是假装不存在
#
# 当前实现匹配的是三句固定话术。模型若换一种说法（"材料中没有相关信息"），
# 就判不出来 —— 因为 3B 小模型的结构化输出（让模型自己标 is_no_info 字段）
# 实测不可靠（会给字段填反），所以暂时接受这个风险。详见 README「实验记录」。
#
# 用 xfail 而不是删掉：哪天换了更强的模型 / 加了 cross-encoder 重排，
# 把这条改成正常断言，它能立刻告诉你"这个限制已经解决了"。
# ============================================================
@pytest.mark.xfail(
    reason="已知限制：模型改说「材料中没有相关信息」时判不出（3B 模型结构化输出不可靠，见 README）",
    strict=True,
)
def test_known_limitation_other_wording():
    assert is_no_info_answer("材料中没有相关信息。") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
