# -*- coding: utf-8 -*-
"""
文档加载与清洗模块（doc_loader）
================================
支持格式：.txt / .md / .pdf
PDF 处理流程：提取文本 → 判断是否扫描件 → 删页眉页脚 → 合并断行 → 清理噪声

对外接口：
    load_document(path) -> (文件名, 干净文本, 元信息字典)
"""

import os
import re

from pypdf import PdfReader

# ---------- 配置 ----------
SCANNED_AVG_CHARS = 20      # 平均每页字符数低于此值 → 疑似扫描件
END_PUNCT = ('。', '！', '!', '?', '？', ':', '：', ';', '；')
LIST_PREFIX = ('•', '-', '*')
TITLE_PAT = re.compile(r"^(\d+[\.、]|第[一二三四五六七八九十]+[章节])")


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


def remove_repeated_lines(pages):
    """删除在 >=2 个页面中重复出现的行（页眉页脚）"""
    line_page_count = {}
    for page in pages:
        for line in set(page.split("\n")):          # set：同一页内重复只算一次
            if not line.strip():                    # 空行不参与统计
                continue
            key = normalize(line)
            line_page_count[key] = line_page_count.get(key, 0) + 1

    repeated_keys = {k for k, c in line_page_count.items() if c >= 2}

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
