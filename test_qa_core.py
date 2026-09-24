# -*- coding: utf-8 -*-
"""
问答核心回归测试
================
跑法：python test_qa_core.py

覆盖"模型拒答判定"（qa_core.is_no_info_answer）的边界用例。
纯函数测试，不需要 Ollama，秒级跑完。每次改拒答判定或提示词都跑一遍。

背景：拒答判定踩过两个坑，这里的用例就是为它们锁的
  1. 用短词表 ("没有找到", "没有相关", ...) 匹配
     → 把含"没有找到键"的正常回答误判成拒答，来源被错误清空
  2. 直接"精确等于固定话术"
     → 模型实测会丢掉句末标点，导致漏判
"""

from qa_core import is_no_info_answer

passed = 0
failed = 0


def check(name, condition):
    """断言辅助：condition 为 True 通过"""
    global passed, failed
    if condition:
        passed += 1
        print(f"✅ {name}")
    else:
        failed += 1
        print(f"❌ {name}")


# 执行部分放在 __main__ 里：pytest 导入本模块时不会运行这些用例，
# 因此 `pytest` 不会因为模块级的 SystemExit 报错；直接运行本文件则照常执行。
if __name__ == "__main__":
    # ---------- 用例组 1：应判为「拒答」→ 来源要清空 ----------
    should_refuse = [
        ("标准话术（带句号）", "知识库中没有找到相关信息。"),
        ("标准话术（无句号——模型实测会丢标点）", "知识库中没有找到相关信息"),
        ("标准话术 + 附带建议", "知识库中没有找到相关信息。建议补充相关文档。"),
        ("标准话术 + 空格换行干扰", "知识库中没有找到\n相关信息。"),
        ("变体：缺「知识库中」", "没有找到相关信息。"),
        ("变体：没有「找到」", "知识库中没有相关信息。"),
    ]
    for name, answer in should_refuse:
        check("拒答 → " + name, is_no_info_answer(answer) is True)

    # ---------- 用例组 2：不应判为「拒答」→ 来源要保留 ----------
    # 这几条是回归用例：旧的短词表会把它们全部误判成拒答
    should_keep = [
        ("含「没有找到」但是有效回答",
         "dict.get() 可以在键不存在时返回 None，避免'没有找到键'的报错。"),
        ("含「未找到」但是有效回答",
         "字典取值时如果键未找到，用 get() 返回默认值。"),
        ("含「无法确定」但是有效回答",
         "无法确定具体型号时，可参考手册中的 S7-1200 参数表。"),
        ("普通有效回答",
         "RAG 代表检索增强生成（Retrieval-Augmented Generation）。"),
    ]
    for name, answer in should_keep:
        check("保留 → " + name, is_no_info_answer(answer) is False)

    # ---------- 已知残留风险（留作记录，不参与断言）----------
    # 窄匹配的代价：模型若换用别的说法就判不出来。结构化输出方案实测 3B 模型不可靠
    # （会给字段填反），所以暂时接受这个风险（详见 README「实验记录」）。
    residual = "材料中没有相关信息。"
    print(f"\n已知残留：模型若改说「材料中未提及」这类别的措辞会漏判"
          f"（当前 is_no_info_answer({residual!r}) = {is_no_info_answer(residual)}）")

    # ---------- 结果 ----------
    print(f"\n通过 {passed} / {passed + failed}")
    raise SystemExit(1 if failed else 0)
