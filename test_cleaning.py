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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
