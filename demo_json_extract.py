# -*- coding: utf-8 -*-
"""
演示：从模型输出里提取 JSON 的三种方式对比
==========================================
跑法：python demo_json_extract.py
"""

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


def greedy(text):
    """① 贪婪匹配：.* 默认匹配尽可能长 → 从第一个 { 到最后一个 }"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def non_greedy(text):
    """② 非贪婪匹配：.*? 匹配尽可能短 → 从第一个 { 到第一个 }"""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def balanced(text):
    """③ 括号配平：从每个 { 开始找与它配对的 }，逐个尝试解析"""
    try:
        return json.loads(text)
    except Exception:
        pass

    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


CASES = [
    ("多个 JSON 对象（模型既输出工具调用又输出回答）",
     '{"action": "tool", "tool": "list_documents", "args": {}}\n'
     '{"action": "answer", "content": "共 3 个文档"}'),

    ("嵌套 JSON（工具参数本身是对象）",
     '{"action": "tool", "args": {"device_id": "S7-1200-01"}}'),

    ("JSON 前后夹带解释文字",
     '好的，我来查一下：\n{"action": "tool", "tool": "doc_stats", "args": {}}\n以上是决策结果'),
]

for title, text in CASES:
    print("=" * 66)
    print(f"场景：{title}")
    print(f"模型输出：{text!r}")
    print("-" * 66)
    for name, func in (("① 贪婪  .*", greedy),
                       ("② 非贪婪 .*?", non_greedy),
                       ("③ 括号配平", balanced)):
        result = func(text)
        mark = "✅" if result else "❌"
        print(f"  {mark} {name:<12} → {result}")
    print()

print("=" * 66)
print("结论")
print("=" * 66)
print("  ① 贪婪  .*   → 匹配【太多】：多个 JSON 被黏在一起 → 解析失败")
print("  ② 非贪婪 .*?  → 匹配【太少】：嵌套 JSON 被提前截断 → 解析失败")
print("  ③ 括号配平    → 正好：数括号深度，找到【配对的那个 }】")
print("                  —— 既能处理嵌套，又能处理多个对象，还能忽略前后废话")
