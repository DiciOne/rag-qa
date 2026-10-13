# -*- coding: utf-8 -*-
"""
Agent 核心：把项目已有能力包装成工具，让模型自己决定调用哪个
==============================================================
工具集：
  - search_manual(query)  → RAG 语义检索（hybrid_retriever + Chroma）
  - list_documents()      → 列出知识库文档（doc_store / SQLite）
  - doc_stats()           → 知识库统计（doc_store / SQLite）

工作方式：**ReAct 循环** —— 决策 → 执行工具 → 观察结果 → 再决策，最多 max_steps 步。

设计要点：
  1. 工具只返回"素材"，不返回"生成好的答案"
     —— Agent 会基于素材推理；工具里再做一次生成既慢，又会让信息被压缩丢失。
  2. 工具注册表 + 动态分发 —— 新增工具只需注册，不用改调用逻辑。
  3. 全程带过程日志（verbose）—— 便于调试"模型是在哪一步判断错的"。
"""

import json
import re

from qa_core import ask, ModelServiceError

CHAT_MODEL = "qwen2.5:3b"
MAX_STEPS = 4          # 最多循环几步（防止无限调工具）


def parse_json(text):
    """从模型输出里解析出**第一个完整的 JSON 对象**

    模型经常不老实：可能在 JSON 前后夹带解释文字，也可能一次输出多个 JSON。
    所以不能简单用 re.search(r"\\{.*\\}") —— 那个正则会贪婪匹配到最后一个 }，
    把多个 JSON 黏在一起导致解析失败。

    这里用"括号配平"的方式：从每个 { 开始找与它配对的 }，逐个尝试解析。
    """
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
                        break          # 这一段不是合法 JSON，换下一个 { 试
    return None


class Agent:
    """基于工具调用的智能体（ReAct 循环）"""

    def __init__(self, engine, store=None, max_steps=MAX_STEPS, verbose=True):
        self.engine = engine          # QAEngine（RAG）
        self.store = store            # DocStore（SQL），可以为 None
        self.max_steps = max_steps
        self.verbose = verbose

        self.tools = {}               # 工具名 → 函数（注册表）
        self._tool_desc = []          # 工具说明（写进提示词，模型靠它知道有哪些工具）
        self._register_tools()

    # ---------------- 工具注册 ----------------
    def _register(self, name, func, desc):
        self.tools[name] = func
        self._tool_desc.append(f"- {desc}")

    def _register_tools(self):
        self._register(
            "search_manual", self._search_manual,
            "search_manual(query)：检索知识库文档，用于回答「是什么原理」「怎么处理」这类知识问题")

        if self.store is not None:
            self._register(
                "list_documents", self._list_documents,
                "list_documents()：列出知识库里所有文档（名字 / 类型 / 块数 / 字符数），不需要参数")
            self._register(
                "doc_stats", self._doc_stats,
                "doc_stats()：知识库总览统计（文档数 / 块数 / 总字符数），不需要参数")

    # ---------------- 具体工具实现 ----------------
    def _search_manual(self, query):
        """检索知识库：只返回原文片段，不做生成"""
        hits = self.engine.retriever.search(query, k=3)
        if not hits:
            return {"结果": "知识库中未找到相关内容"}
        return {
            "内容": [chunk for chunk, _, _ in hits],
            "来源": list(dict.fromkeys(src for _, src, _ in hits)),
        }

    def _list_documents(self):
        return {"文档列表": self.store.list_documents()}

    def _doc_stats(self):
        return {"统计": self.store.stats()}

    # ---------------- 提示词 ----------------
    def _build_prompt(self, question, scratchpad):
        return (
            "你是一个知识库运维助手。你可以调用下面这些工具来获取信息：\n"
            + "\n".join(self._tool_desc) + "\n\n"
            "【已经执行过的步骤】\n"
            + (scratchpad if scratchpad else "（还没有执行任何步骤）") + "\n\n"
            "【决策规则】\n"
            "1. 如果需要更多信息，输出：\n"
            '   {"action": "tool", "tool": "工具名", "args": {"参数名": "参数值"}}\n'
            "2. 如果已经能回答用户问题，输出：\n"
            '   {"action": "answer", "content": "你的回答"}\n'
            "3. **凡是涉及知识库内容的事实性问题**（有哪些文档、多少块、文档里写了什么、"
            "故障代码含义、处理方法），**必须先调用工具获取**，绝不能凭自己的知识回答或编造\n"
            "4. answer 里的 content **必须是给用户看的自然语言句子**，不要是 JSON 或字典格式\n"
            "5. 同一个工具不要用相同参数重复调用\n"
            "6. 只输出 JSON，不要任何其他文字\n\n"
            "【示例】（照着这些例子的格式做）\n"
            '用户：知识库里有哪些文档？\n'
            '输出：{"action": "tool", "tool": "list_documents", "args": {}}\n\n'
            '用户：知识库一共多少块？\n'
            '输出：{"action": "tool", "tool": "doc_stats", "args": {}}\n\n'
            '用户：S7-1200 温度过高怎么处理？\n'
            '输出：{"action": "tool", "tool": "search_manual", "args": {"query": "温度过高 处理方法"}}\n\n'
            '用户：你好\n'
            '输出：{"action": "answer", "content": "你好！我是知识库运维助手，可以帮你查询知识库内容和文档信息。"}\n\n'
            "用户问题：" + question + "\n"
        )

    # ---------------- 执行工具 ----------------
    def _execute(self, decision):
        tool_name = decision.get("tool")
        args = decision.get("args", {}) or {}
        func = self.tools.get(tool_name)
        if func is None:
            return {"错误": f"未知工具 {tool_name}"}
        try:
            return func(**args)
        except TypeError as e:
            return {"错误": f"工具参数不对：{e}"}

    # ---------------- 主循环 ----------------
    def run(self, question):
        """ReAct 循环：决策 → 执行工具 → 观察 → 再决策"""
        scratchpad = ""
        for step in range(self.max_steps):
            reply = ask(self._build_prompt(question, scratchpad))
            decision = parse_json(reply)

            if decision is None:
                if self.verbose:
                    print(f"  [第{step + 1}步] 模型输出无法解析，原文：{reply.strip()[:120]}")
                return "抱歉，我没理解你的问题"

            action = decision.get("action")

            if action == "answer":
                if self.verbose:
                    print(f"  [第{step + 1}步] 决策 → 直接回答")
                return decision.get("content", "")

            if action == "tool":
                tool_name = decision.get("tool")
                args = decision.get("args", {}) or {}
                if self.verbose:
                    print(f"  [第{step + 1}步] 决策 → 调用工具 {tool_name}，参数 {args}")

                result = self._execute(decision)
                preview = json.dumps(result, ensure_ascii=False)
                if self.verbose:
                    print(f"           工具返回 → {preview[:80]}{'…' if len(preview) > 80 else ''}")

                scratchpad += (
                    f"\n第{step + 1}步：调用 {tool_name}，参数 {args}\n"
                    f"返回结果：{preview}\n"
                )
                continue

            if self.verbose:
                print(f"  [第{step + 1}步] 决策异常（action={action!r}），放弃")
            return "抱歉，我没理解你的问题"

        return "抱歉，我尝试了多次仍无法完成"
