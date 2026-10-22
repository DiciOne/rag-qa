# -*- coding: utf-8 -*-
"""
文档加载与清洗模块（doc_loader）
================================
支持格式：.txt / .md / .pdf
PDF 处理流程：提取文本 → 判断是否扫描件 → 删页眉页脚 → 合并断行 → 清理噪声

对外接口：
    load_document(path) -> (文件名, 干净文本, 元信息字典)
"""

import math
import os
import re

from pypdf import PdfReader

import config

# ---------- 配置 ----------
SCANNED_AVG_CHARS = 20      # 平均每页字符数低于此值 → 疑似扫描件
END_PUNCT = ('。', '！', '!', '?', '？', ':', '：', ';', '；')
LIST_PREFIX = ('•', '-', '*')
TITLE_PAT = re.compile(r"^(\d+[\.、]|第[一二三四五六七八九十百零]+[章节])")


# ============================================================
# 一、PDF 文本提取
# ============================================================
def extract_pdf_pages(path):
    """提取 PDF 每页文本，返回 (页面文本列表, 是否疑似扫描件)

    注意：Chroma 提取时空白页返回 None，这里统一转成空字符串，
         保持"列表索引 == 页码"的对应关系。
    """
    reader = PdfReader(path)
    pages = []
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text() or ""
        pages.append(text)

    total_chars = sum(len(p) for p in pages)
    avg = total_chars / len(pages) if pages else 0
    is_scanned = avg < SCANNED_AVG_CHARS
    return pages, is_scanned


# ============================================================
# 二、文本清洗
# ============================================================
# 只归一化"页码"这类模式，**不要**把所有数字都替换掉。
# 踩坑：早期版本用 re.sub(r"\d+", "#", line) 归一化全部数字，结果
#      "工作电压：DC 24V" 和 "工作电压：DC 12V" 变成同一个 key，
#      被误判为"重复行"而整行删除（真实手册的参数表会被成片删掉）。
PAGE_NUM_PAT = re.compile(r"第\s*\d+\s*页|Page\s*\d+|^\s*\d+\s*$", re.IGNORECASE)


def normalize(line):
    """归一化页码类内容，用于识别"带页码的页脚"这类重复行"""
    return PAGE_NUM_PAT.sub("#", line)


def remove_repeated_lines(pages, min_ratio=None):
    """删除"几乎每页都出现"的行（页眉页脚）

    ★ 为什么不能只看"重复 2 次就删"
    ------------------------------------------------------------
    第一版规则是"某行在 >=2 个页面出现 → 当页眉删掉"。在小示例上没问题，
    上了真实手册立刻崩：

      台达 DVP 手册（976 页）：删掉了 30.8% 的正文
        误删的例子："3. 指令集"（318 页）、"運算元"（239 页，表格表头）、
                    "X0"（321 页，指令操作数）
      西门子 S7-1200 手册（329 页）：删掉了 32.1% 的正文
        误删的例子："技术规范"（97 页）、"4.4 程序结构"（48 页）

    根因：**章节标题、表格表头、章节页眉天生就在大量页面重复**，
    和真页眉页脚长得一模一样，光看"出现过几次"区分不了。

    ★ 修法：加一个「出现比例」门槛
    ------------------------------------------------------------
    真页眉页脚几乎每页都有，而章节标题/表头只在它所在的那几十页重复。
    实测比例（来自上面两本真手册）：

      "S7-1200 G2 可编程逻辑控制器"（真页眉）→ 99.7%   ✅ 该删
      "3. 指令集"（章节页眉）                → 32.6%   ❌ 不能删
      "技术规范"（章节名）                    → 29.5%   ❌ 不能删
      "運算元"（表格表头）                    → 24.5%   ❌ 不能删
      "X0"（正文）                            → 32.9%   ❌ 不能删

    所以取 50% 作为门槛：真页眉留下，所有误删候选全被排除。

    ★ 为什么宁可漏删也不误删
    ------------------------------------------------------------
    漏删一页页眉 = 多几行噪声（影响很小，BM25 里几个短词而已）。
    误删正文   = 内容永久丢失，而且下游检索和回答全都跟着变差。
    两者代价不对称，所以阈值定得保守。实测确实会漏掉一些"章节页眉"
    （如台达的 "3. 指令集" 32.6%），这是有意接受的取舍。

    ★ 为什么不用"行的位置"来判断
    ------------------------------------------------------------
    试过：真页眉理论上固定在页面首尾。但 pypdf 的 extract_text() 按
    内容流顺序输出、不是视觉顺序，实测同一行页眉在 329 页里有 149 页
    被排到了"中间"，位置信号不可靠。比例信号则非常干净（99.7% vs 32.6%）。
    """
    if min_ratio is None:
        min_ratio = config.PDF_REPEAT_MIN_RATIO

    total_pages = len(pages)
    if total_pages == 0:
        return []

    # 每个归一化行 → 出现过它的页数（同一页内重复只算一次）
    page_count = {}
    for page in pages:
        for key in {normalize(line) for line in page.split("\n") if line.strip()}:
            page_count[key] = page_count.get(key, 0) + 1

    # 门槛：至少 2 页（单页重复不算页眉），且占比 >= min_ratio
    threshold = max(2, math.ceil(total_pages * min_ratio))
    repeated_keys = {k for k, c in page_count.items() if c >= threshold}

    cleaned_pages = []
    for page in pages:
        kept_lines = []
        for line in page.split("\n"):
            if line.strip() and normalize(line) in repeated_keys:
                continue                            # 页眉页脚：丢弃
            kept_lines.append(line)
        cleaned_pages.append("\n".join(kept_lines))
    return cleaned_pages


def should_break(current, nxt):
    """判断 current 之后是否应该断开（不与该行合并）"""
    if current == "":
        return True                              # 空行
    if TITLE_PAT.match(current):
        return True                              # 标题行（1. xxx / 第X章）
    if current.endswith(END_PUNCT):
        return True                              # 句子已结束
    if nxt.startswith(LIST_PREFIX) or TITLE_PAT.match(nxt):
        return True                              # 下一行是新条目
    return False


def merge_broken_lines(text):
    """合并被 PDF 排版硬切断的行（缓冲区算法）"""
    lines = text.split("\n")
    result, buffer = [], ""
    for line in lines:
        if buffer == "":
            buffer = line                        # 手上没东西，先拿着
        elif should_break(buffer, line):
            result.append(buffer)                # 该断：交出去
            buffer = line
        else:
            buffer = buffer + line               # 该接：拼起来继续等
    if buffer != "":
        result.append(buffer)
    return "\n".join(result)


def clean_text(text):
    """清理多余空格、特殊字符、行首尾空白"""
    text = re.sub(r" {2,}", " ", text)           # 连续空格压缩成一个
    text = text.replace("\ufb03", "fi")          # 连字字符 ﬁ
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def clean_pages(pages):
    """完整清洗流水线：删页眉页脚 → 合并断行 → 清噪声"""
    pages = remove_repeated_lines(pages)
    cleaned = [clean_text(merge_broken_lines(p)) for p in pages]
    return "\n".join(cleaned)


# ============================================================
# 三、统一入口
# ============================================================
def load_document(path):
    """按扩展名加载文档，返回 (文件名, 干净文本, 元信息)"""
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    meta = {"path": path, "type": ext.lstrip(".")}

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        meta["pages"] = 1
        meta["is_scanned"] = False
        return name, text, meta

    if ext == ".pdf":
        pages, is_scanned = extract_pdf_pages(path)
        text = clean_pages(pages)
        meta["pages"] = len(pages)
        meta["is_scanned"] = is_scanned
        if is_scanned:
            print(f"  ⚠️  {name} 疑似扫描件（文字层为空），需要 OCR 才能入库")
        return name, text, meta

    return name, "", meta


def supported_extensions():
    return (".txt", ".md", ".pdf")
