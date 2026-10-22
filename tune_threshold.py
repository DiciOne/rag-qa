# -*- coding: utf-8 -*-
"""
阈值标定工具（tune_threshold）
================================
跑法：python tune_threshold.py

为什么需要这个工具
------------------
`MAX_DISTANCE` 是"该不该拒答"的判据，但它**不是常数** —— 它随知识库规模漂移。

实测记录：
    18 块的小库（3 个文档）       → 相关 ≈ 0.39，无关 ≈ 0.59，最优阈值 0.5
    2822 块（西门子手册 329 页）   → 最优阈值 0.44
    11048 块（+ 台达手册 976 页）  → 见本工具输出

原因不难理解：库越大，"最相似的邻居"就越容易恰好离得近一点，所有距离整体前移。
所以**每次知识库有实质变化（换文档、加手册），都应该重跑一次这个工具**。

它做什么
--------
1. 对评测集里每个问题，取向量检索的 top1 距离
2. 扫 0.30~0.60，统计每个阈值下的
     误拒 = 库内问题被拒（该答却没答）
     误答 = 库外问题没被拒（不该答却答了）
3. 输出总错误数最小的阈值，以及稳定区间（多个阈值并列最优时）

注意：这只是"在现有评测集上最优"。样本有限，别把小数点当真理 ——
真正的解法是接 cross-encoder reranker 做精排判相关（见 README 后续优化）。
"""

import sys

import chromadb
from chromadb.utils import embedding_functions

import config
from eval_retrieval import EVAL_SET
from hybrid_retriever import HybridRetriever

SCAN_FROM = 0.30
SCAN_TO = 0.60
SCAN_STEP = 0.01


def main():
    ef = embedding_functions.OllamaEmbeddingFunction(
        url=config.EMBED_URL, model_name=config.EMBED_MODEL
    )
    col = chromadb.PersistentClient(path=config.CHROMA_DIR).get_collection(
        config.COLLECTION_NAME, embedding_function=ef
    )
    r = HybridRetriever(col, max_distance=config.MAX_DISTANCE)

    print(f"知识库：{col.count()} 块")
    print(f"当前阈值：{config.MAX_DISTANCE}")
    print()

    rows = []
    for item in EVAL_SET:
        vec = r.vector_search(item["q"], 3)
        d1 = vec[0][2] if vec else 9.99
        rows.append((item["q"], item["source"] is not None, d1))

    print("=" * 68)
    print(f"{'问题':<30} {'类型':<6} {'top1距离':>9}")
    print("-" * 68)
    for q, is_in, d1 in sorted(rows, key=lambda x: x[2]):
        print(f"{q[:28]:<30} {'库内' if is_in else '库外':<6} {d1:>9.3f}")

    in_d = [d for _, is_in, d in rows if is_in]
    out_d = [d for _, is_in, d in rows if not is_in]
    print()
    print(f"库内 top1 距离：{min(in_d):.3f} ~ {max(in_d):.3f}   （共 {len(in_d)} 条）")
    print(f"库外 top1 距离：{min(out_d):.3f} ~ {max(out_d):.3f}   （共 {len(out_d)} 条）")

    print()
    print("=" * 68)
    print("阈值扫描（误拒 = 库内被拒；误答 = 库外没拒）")
    print("=" * 68)
    print(f"{'阈值':>6} {'误拒':>5} {'误答':>5} {'总错误':>7}")
    results = []
    t = SCAN_FROM
    while t <= SCAN_TO + 1e-9:
        t = round(t + SCAN_STEP, 2)
        fr = sum(1 for d in in_d if d > t)
        fa = sum(1 for d in out_d if d <= t)
        results.append((t, fr + fa, fr, fa))
        print(f"{t:>6.2f} {fr:>5} {fa:>5} {fr + fa:>7}")

    best_err = min(x[1] for x in results)
    bests = [x[0] for x in results if x[1] == best_err]
    print()
    print(f"最小总错误：{best_err}")
    print(f"达到它的阈值：{bests[0]:.2f} ~ {bests[-1]:.2f}"
          f"（共 {len(bests)} 个，{'区间取中点更稳' if len(bests) > 1 else '单点'}）")
    if len(bests) > 1:
        mid = round((bests[0] + bests[-1]) / 2, 2)
        print(f"建议取：{mid:.2f}")

    # 指出仍有错误的具体问题
    for t in (bests[0], bests[-1]):
        bad_in = [q for q, is_in, d in rows if is_in and d > t]
        bad_out = [q for q, is_in, d in rows if not is_in and d <= t]
        if bad_in or bad_out:
            print()
            print(f"阈值 {t:.2f} 下仍出错：")
            for q in bad_in:
                print(f"  ❌ 误拒（库内被拒）：{q}")
            for q in bad_out:
                print(f"  ❌ 误答（库外没拒）：{q}")
            break


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
