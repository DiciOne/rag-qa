# -*- coding: utf-8 -*-
"""
文档清洗回归测试
================
跑法：python test_cleaning.py

这些是修 bug 时沉淀下来的用例。每次改清洗逻辑都跑一遍，
确保不会把已经修好的问题改回去（这就是"回归测试"的作用）。
"""

from doc_loader import remove_repeated_lines

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
    # ---------- 用例 1：页眉页脚应被删除，正文保留 ----------
    pages = [
        "S7-1200 技术手册\n技术手册  第 1 页\n1. 产品概述\n工作电压：DC 24V",
        "S7-1200 技术手册\n技术手册  第 2 页\n2. 通信设置\nIP 地址必须与 PLC 一致",
    ]
    result = remove_repeated_lines(pages)
    check("页眉（两页完全相同的行）被删除", "S7-1200 技术手册" not in result[0])
    check("带页码的页脚被删除（第 1 页）", "第 1 页" not in result[0])
    check("带页码的页脚被删除（第 2 页）", "第 2 页" not in result[1])
    check("正文（含数字）被保留", "工作电压：DC 24V" in result[0])
    check("正文（含数字）被保留（第 2 页）", "IP 地址必须与 PLC 一致" in result[1])

    # ---------- 用例 2（回归用例）：参数表数值不同时不能被误删 ----------
    # 背景：早期版本把所有数字归一化成 #，导致
    #      "工作电压：DC 24V" 与 "DC 12V" 变同一个 key 而被整行删除
    pages2 = [
        "设备参数表\n工作电压：DC 24V\n额定电流：2A",
        "设备参数表\n工作电压：DC 12V\n额定电流：3A",
    ]
    result2 = remove_repeated_lines(pages2)
    check("参数行未被误删（第 1 页·电压）", "工作电压：DC 24V" in result2[0])
    check("参数行未被误删（第 1 页·电流）", "额定电流：2A" in result2[0])
    check("参数行未被误删（第 2 页·电压）", "工作电压：DC 12V" in result2[1])
    check("参数行未被误删（第 2 页·电流）", "额定电流：3A" in result2[1])

    # ---------- 结果 ----------
    print(f"\n通过 {passed} / {passed + failed}")
    raise SystemExit(1 if failed else 0)
