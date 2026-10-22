# -*- coding: utf-8 -*-
"""
PDF 提取质量检查器（check_extract）
====================================
用法：
    python check_extract.py                # 检查 knowledge/ 下所有文档
    python check_extract.py 某个手册.pdf    # 只检查一个文件

为什么需要它
------------
PDF 提取是会静默出错的：扫描页提取到 0 个字、页眉删不干净、
表格被压成一行、断行合并把两段拼成一段 —— 这些**不会报错**，
只会让后面的检索和回答变差，而你完全看不出来。

这个工具把"看不出来的问题"变成具体数字：

  1. 哪些页提取不到文字（扫描件/纯图页）→ 需要 OCR
  2. 清洗删掉了多少字（删除率过高 = 误删风险，过低 = 漏删）
  3. 清洗后仍跨页重复的行 → 页眉页脚没删干净
  4. 重复行的长度分布 → 长行重复更可能是**正文被误删**
  5. 短行 / 多空格行的比例 → 断行与表格被压扁的痕迹

第 3、4 条是重点：它能同时暴露"漏删"和"误删"这两种相反的 bug。
之前踩过的坑（把所有数字归一化成 #，导致整页参数表被删）如果早跑这个
工具，会看到"删除率异常高 + 重复行里有大量长行"，一眼就能定位。

不需要 Ollama，纯本地分析。
"""

import math
import os
import re
import sys
from collections import Counter

import config
import doc_loader

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

# 判定阈值（按实际情况可调）
EMPTY_PAGE_CHARS = 20      # 低于这个字符数 → 当作空页/图片页
SHORT_LINE_CHARS = 10      # 低于这个长度 → 短行（断行/表格痕迹）
LONG_REPEAT_CHARS = 40     # 重复行超过这个长度 → 更可能是正文被误删


def _short(s, n=46):
    """截断长行用于展示"""
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def analyze_pdf(path):
    """分析一个 PDF：提取分布 + 清洗前后对比"""
    name = os.path.basename(path)

    print("=" * 72)
    print(f"文件：{name}")
    print("=" * 72)

    # ---------- 1. 原始提取 ----------
    try:
        pages, is_scanned = doc_loader.extract_pdf_pages(path)
    except Exception as exc:
        print(f"  ❌ 提取失败：{type(exc).__name__}: {exc}")
        return

    page_chars = [len(p) for p in pages]
    total = sum(page_chars)
    n_pages = len(pages)
    avg = total / n_pages if n_pages else 0

    print(f"  页数        {n_pages}")
    print(f"  提取字符    {total:,}")
    print(f"  平均每页    {avg:.0f} 字")

    if is_scanned:
        print("  ⚠️  整份文件疑似扫描件（文字层几乎为空）→ 需要 OCR 才能入库")
        return

    # ---------- 2. 空页 / 异常页 ----------
    empty = [i + 1 for i, c in enumerate(page_chars) if c < EMPTY_PAGE_CHARS]
    if empty:
        print(f"  ⚠️  提取不到文字的页：{len(empty)} 页 "
              f"({len(empty) / n_pages * 100:.1f}%) → 扫描页或纯图片页")
        print(f"      页号（最多列 20 个）：{empty[:20]}")
    else:
        print("  ✅ 没有空白页")

    if page_chars:
        order = sorted(range(n_pages), key=lambda i: page_chars[i])
        least = [(i + 1, page_chars[i]) for i in order[:3]]
        most = [(i + 1, page_chars[i]) for i in order[-3:]][::-1]
        print(f"  字符最少的页  {', '.join(f'第{p}页({c}字)' for p, c in least)}")
        print(f"  字符最多的页  {', '.join(f'第{p}页({c}字)' for p, c in most)}")

    # ---------- 3. 重复行（页眉页脚候选）----------
    counter = Counter()
    for page in pages:
        for line in set(page.split("\n")):        # 同页内重复只算一次
            if line.strip():
                counter[doc_loader.normalize(line)] += 1

    repeated = {k: c for k, c in counter.items() if c >= 2}
    covered_pages = sum(repeated.values())
    threshold_pre = max(2, math.ceil(n_pages * config.PDF_REPEAT_MIN_RATIO))
    doomed = {k: c for k, c in repeated.items() if c >= threshold_pre}
    print()
    print(f"  跨页重复行    {len(repeated)} 种，合计覆盖 {covered_pages} 页次")
    print(f"               其中会达到门槛被删掉：{len(doomed)} 种"
          f"（门槛 {threshold_pre} 页）；其余低于门槛，保留")

    if repeated:
        # 按长度分档：短行像页眉，长行像正文被误删 —— 但只有"会被删的"才危险
        buckets = {"短行(<=15字)": 0, f"中行(16-{LONG_REPEAT_CHARS}字)": 0,
                   f"长行(>{LONG_REPEAT_CHARS}字)": 0}
        for k in doomed:
            ln = len(k)
            if ln <= 15:
                buckets["短行(<=15字)"] += 1
            elif ln <= LONG_REPEAT_CHARS:
                buckets[f"中行(16-{LONG_REPEAT_CHARS}字)"] += 1
            else:
                buckets[f"长行(>{LONG_REPEAT_CHARS}字)"] += 1
        print("      会被删的行长度分布：" + " | ".join(f"{k} {v}" for k, v in buckets.items()))
        if buckets[f"长行(>{LONG_REPEAT_CHARS}字)"] > 0:
            print("      ⚠️  有长行会被删 → 可能是正文/参数表被误判成页眉页脚")

        print("      Top 8（按出现页数）：")
        for line, cnt in sorted(repeated.items(), key=lambda x: -x[1])[:8]:
            mark = "删除" if cnt >= threshold_pre else "保留"
            print(f"        {cnt:>4} 页 [{mark}]  {_short(line)}")

    # ---------- 4. 清洗前后对比 ----------
    cleaned_pages = doc_loader.remove_repeated_lines(pages)
    cleaned_total = sum(len(p) for p in cleaned_pages)
    removed = total - cleaned_total
    rate = removed / total * 100 if total else 0

    # 当前生效的判定门槛（跟 doc_loader 用同一个配置，保证结论一致）
    threshold = max(2, math.ceil(n_pages * config.PDF_REPEAT_MIN_RATIO))

    print()
    print("  —— 清洗（删页眉页脚）前后 ——")
    print(f"      判定门槛  某行需出现在 >={threshold} 页"
          f"（占 {config.PDF_REPEAT_MIN_RATIO:.0%}）才当页眉删掉")
    print(f"      清洗前  {total:,} 字")
    print(f"      清洗后  {cleaned_total:,} 字")
    print(f"      删除    {removed:,} 字（{rate:.1f}%）")

    if rate > 15:
        print("      ⚠️  删除率偏高（>15%）→ 有误删正文的风险，建议抽查")
    elif rate < 0.5 and repeated:
        print("      ℹ️  删除率很低 —— 说明这本手册的页眉页脚出现得不密集，"
              "被阈值挡在了外面（漏删，不丢内容）")
    else:
        print("      ✅ 删除率在合理范围")

    # 清洗后仍然重复的行：区分「低于阈值有意保留」和「真的漏删」
    after = Counter()
    for page in cleaned_pages:
        for line in set(page.split("\n")):
            if line.strip():
                after[doc_loader.normalize(line)] += 1
    leftover = {k: c for k, c in after.items() if c >= 2}
    if leftover:
        below = {k: c for k, c in leftover.items() if c < threshold}
        real_leak = {k: c for k, c in leftover.items() if c >= threshold}
        print(f"      清洗后仍跨页重复：{len(leftover)} 种"
              f"（其中低于门槛有意保留 {len(below)} 种）")
        if real_leak:
            print(f"      ⚠️  达到门槛却没被删：{len(real_leak)} 种 → 真漏删，抽查：")
            for line, cnt in sorted(real_leak.items(), key=lambda x: -x[1])[:5]:
                print(f"        {cnt:>4} 页  {_short(line)}")
        else:
            print("      ✅ 达到门槛的重复行都被删掉了（没有真漏删）")
        if below:
            print("      有意保留的 Top 3（避免误删章节标题/表头）：")
            for line, cnt in sorted(below.items(), key=lambda x: -x[1])[:3]:
                print(f"        {cnt:>4} 页  {_short(line)}")
    else:
        print("      ✅ 清洗后没有残留的跨页重复行")

    # ---------- 5. 断行 / 表格痕迹 ----------
    all_lines = [ln.strip() for p in cleaned_pages for ln in p.split("\n") if ln.strip()]
    if all_lines:
        short = sum(1 for ln in all_lines if len(ln) <= SHORT_LINE_CHARS)
        multi_space = sum(1 for ln in all_lines if re.search(r" {2,}", ln))
        digit_heavy = sum(
            1 for ln in all_lines
            if len(re.findall(r"\d", ln)) >= 4 and re.search(r"\s", ln)
        )
        print()
        print("  —— 排版痕迹（清洗后）——")
        print(f"      总行数            {len(all_lines):,}")
        print(f"      短行(<=10字)      {short:,} ({short / len(all_lines) * 100:.1f}%)"
              f"   ← 断行或被拆散的表格")
        print(f"      含连续空格的行    {multi_space:,} "
              f"({multi_space / len(all_lines) * 100:.1f}%)   ← 表格被压成一行")
        print(f"      多数字+含空格的行 {digit_heavy:,} "
              f"({digit_heavy / len(all_lines) * 100:.1f}%)   ← 参数表痕迹")

    # ---------- 6. 最终入模文本规模 ----------
    text = doc_loader.clean_pages(pages)
    chunk_size = config.CHUNK_SIZE
    chunks = (len(text) + chunk_size - 1) // chunk_size
    print()
    print("  —— 最终结果 ——")
    print(f"      入库文本   {len(text):,} 字")
    print(f"      按 {chunk_size} 字切块 ≈ {chunks:,} 块")
    print()


def analyze_text_file(path):
    """txt / md 只做简单统计"""
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print("=" * 72)
    print(f"文件：{name}")
    print("=" * 72)
    print(f"  字符  {len(text):,}")
    print(f"  行数  {len(lines):,}")
    chunk_size = config.CHUNK_SIZE
    print(f"  按 {chunk_size} 字切块 ≈ {(len(text) + chunk_size - 1) // chunk_size:,} 块")
    print()


def main(argv):
    targets = argv[1:]
    if not targets:
        if not os.path.isdir(KNOWLEDGE_DIR):
            print(f"找不到目录：{KNOWLEDGE_DIR}")
            return
        targets = [os.path.join(KNOWLEDGE_DIR, f)
                   for f in sorted(os.listdir(KNOWLEDGE_DIR))
                   if f.lower().endswith(doc_loader.supported_extensions())]

    if not targets:
        print("没有可检查的文件。")
        return

    for path in targets:
        if not os.path.isfile(path):
            print(f"跳过（不是文件）：{path}")
            continue
        if path.lower().endswith(".pdf"):
            analyze_pdf(path)
        elif path.lower().endswith((".txt", ".md")):
            analyze_text_file(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv)
