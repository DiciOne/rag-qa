# -*- coding: utf-8 -*-
"""
文档清洗回归测试
================
跑法：
    pytest test_cleaning.py -v        # 推荐（会被 pytest 自动发现，计入用例统计）
    python test_cleaning.py           # 也可以，直接跑本文件

这些是修 bug 时沉淀下来的用例。每次改清洗逻辑都跑一遍，
确保不会把已经修好的问题改回去（这就是"回归测试"的作用）。
"""

import pytest

from doc_loader import remove_repeated_lines


# ============================================================
# 用例组 1：页眉页脚应被删除，正文保留
# ============================================================
PAGES_WITH_HEADER = [
    "S7-1200 技术手册\n技术手册  第 1 页\n1. 产品概述\n工作电压：DC 24V",
    "S7-1200 技术手册\n技术手册  第 2 页\n2. 通信设置\nIP 地址必须与 PLC 一致",
]


@pytest.mark.parametrize(
    "page_idx, should_be_gone",
    [
        (0, "S7-1200 技术手册"),
        (0, "第 1 页"),
        (1, "第 2 页"),
    ],
    ids=["页眉（两页相同行）被删除", "带页码的页脚被删除（第 1 页）", "带页码的页脚被删除（第 2 页）"],
)
def test_removes_repeated_header_and_footer(page_idx, should_be_gone):
    """重复出现的页眉、带页码的页脚 → 必须删除"""
    result = remove_repeated_lines(PAGES_WITH_HEADER)
    assert should_be_gone not in result[page_idx]


@pytest.mark.parametrize(
    "page_idx, should_be_kept",
    [
        (0, "工作电压：DC 24V"),
        (1, "IP 地址必须与 PLC 一致"),
    ],
    ids=["正文（含数字）被保留", "正文（第 2 页）被保留"],
)
def test_keeps_body_text(page_idx, should_be_kept):
    """正文（哪怕含数字）→ 必须保留，不能被当成页脚删掉"""
    result = remove_repeated_lines(PAGES_WITH_HEADER)
    assert should_be_kept in result[page_idx]


# ============================================================
# 用例组 2（回归用例）：参数表数值不同时不能被误删
#
# 背景：早期版本把所有数字归一化成 #，于是
#       "工作电压：DC 24V" 与 "工作电压：DC 12V" 变成同一个 key，
#       被当成"重复行"整行删除，结果整页参数表被清空。
#       修复方案：只对页码模式归一化（第 N 页 / Page N / 纯数字行）。
# ============================================================
PAGES_WITH_DIFFERENT_VALUES = [
    "设备参数表\n工作电压：DC 24V\n额定电流：2A",
    "设备参数表\n工作电压：DC 12V\n额定电流：3A",
]


@pytest.mark.parametrize(
    "page_idx, should_be_kept",
    [
        (0, "工作电压：DC 24V"),
        (0, "额定电流：2A"),
        (1, "工作电压：DC 12V"),
        (1, "额定电流：3A"),
    ],
    ids=[
        "参数行未被误删（第 1 页·电压）",
        "参数行未被误删（第 1 页·电流）",
        "参数行未被误删（第 2 页·电压）",
        "参数行未被误删（第 2 页·电流）",
    ],
)
def test_keeps_differing_parameter_rows(page_idx, should_be_kept):
    """两页只有数值不同的参数行 → 不是重复行，必须全部保留"""
    result = remove_repeated_lines(PAGES_WITH_DIFFERENT_VALUES)
    assert should_be_kept in result[page_idx]


# ============================================================
# 用例组 3（回归用例）：真实工业手册上的大规模误删
#
# 背景：第一版规则是"某行在 >=2 个页面出现 → 当页眉删掉"。
# 在小示例上完全正常，上了两本真实手册立刻崩：
#
#   台达 DVP 手册（976 页）   → 删掉 30.8% 的正文
#      误删 "3. 指令集"(318 页)、"運算元"(239 页，表格表头)、
#           "X0"(321 页，指令操作数)
#   西门子 S7-1200（329 页）  → 删掉 32.1% 的正文
#      误删 "技术规范"(97 页)、"4.4 程序结构"(48 页)
#
# 根因：章节标题 / 表格表头 / 章节页眉**天生就在大量页面重复**，
#       和真页眉页脚长得一样，光看"重复次数"区分不了。
#
# 修复：改为"出现页数占比 >= 50%"。实测比例：
#   真页眉 "S7-1200 G2 可编程逻辑控制器" → 99.7%  → 删
#   章节标题 "技术规范"                 → 29.5%  → 留
#   表格表头 "運算元"                   → 24.5%  → 留
#   正文     "X0"                      → 32.9%  → 留
# ============================================================

def _make_pages(total, repeated_line, repeat_pages):
    """造 total 页，前 repeat_pages 页都插一行 repeated_line，其余是普通正文"""
    pages = []
    for i in range(total):
        lines = [f"正文第{i + 1}页的第一行内容，需要足够长以免被当成空行"]
        if i < repeat_pages:
            lines.insert(0, repeated_line)
        lines.append(f"正文第{i + 1}页的第二行内容，同样要足够长")
        pages.append("\n".join(lines))
    return pages


def test_removes_true_header():
    """真页眉：几乎每页都有（10/10 = 100%）→ 必须删除"""
    pages = _make_pages(10, "S7-1200 G2 可编程逻辑控制器", 10)
    result = remove_repeated_lines(pages)
    assert all("S7-1200 G2 可编程逻辑控制器" not in p for p in result)


def test_keeps_chapter_title():
    """章节标题：只在 3/10 页出现（30%）→ 不能删，否则丢内容"""
    pages = _make_pages(10, "技术规范", 3)
    result = remove_repeated_lines(pages)
    assert sum(1 for p in result if "技术规范" in p) == 3


def test_keeps_table_header():
    """表格表头：只在 2/10 页出现（20%）→ 不能删"""
    pages = _make_pages(10, "運算元", 2)
    result = remove_repeated_lines(pages)
    assert sum(1 for p in result if "運算元" in p) == 2


def test_ratio_threshold_boundary():
    """门槛边界：恰好 50%（5/10）删；49% 以下（4/10）保留"""
    at_threshold = remove_repeated_lines(_make_pages(10, "边界行A", 5))
    below = remove_repeated_lines(_make_pages(10, "边界行B", 4))
    assert all("边界行A" not in p for p in at_threshold)
    assert sum(1 for p in below if "边界行B" in p) == 4


def test_single_page_occurrence_never_removed():
    """只在 1 页出现的行永远不删（阈值下限是 2 页）"""
    pages = _make_pages(20, "只出现一次的行", 1)
    result = remove_repeated_lines(pages)
    assert sum(1 for p in result if "只出现一次的行" in p) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
