# -*- coding: utf-8 -*-
"""
集中配置
=========
所有可调参数都从这里读，支持 .env 覆盖。

为什么要有这个文件（而不是把参数散落在各模块里）
------------------------------------------------
1. 换环境只改配置，不改代码：本地跑用 localhost，容器里要指向宿主机就用
   OLLAMA_BASE=http://host.docker.internal:11434，一行环境变量解决。
2. 换模型/调阈值不用翻代码：相似度阈值、top_k、切块大小都是"调参"，
   不该和业务逻辑混在一起。
3. 参数有唯一出处：以前 MAX_DISTANCE 在 qa_core、api、retriever 三处各写一份，
   改一处忘一处就会出诡异 bug。

优先级：环境变量  >  .env 文件  >  代码内默认值
（.env 已被 .gitignore 排除，不会把本地地址/密钥提交到仓库）
"""

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 读同目录下的 .env；文件不存在就静默跳过（生产环境通常直接注入环境变量）
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _get(name, default):
    """读字符串配置；空字符串按"没配"处理"""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_int(name, default):
    try:
        return int(_get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name, default):
    try:
        return float(_get(name, default))
    except (TypeError, ValueError):
        return default


# ============================================================
# 模型服务（Ollama）
# ============================================================
OLLAMA_BASE = _get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_URL = _get("OLLAMA_URL", OLLAMA_BASE + "/api/chat")        # 生成
EMBED_URL = _get("EMBED_URL", OLLAMA_BASE + "/api/embeddings")    # 向量化

CHAT_MODEL = _get("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = _get("EMBED_MODEL", "bge-m3")

# 没有超时的话，模型卡住会把请求永久挂起
REQUEST_TIMEOUT = _get_int("REQUEST_TIMEOUT", 120)

# ============================================================
# 检索 / 切块
# ============================================================
# 向量路的相似度阈值。★ 这个值**随知识库规模变化**，库变了必须重新标定。
#
#   18 块的小库（2026-09）：相关 ≈ 0.39，无关 ≈ 0.59 → 当时取 0.5
#   2822 块的真实库（西门子 + 台达手册，2026-10）：
#     用 19 条评测问题实测 top1 距离，扫描 0.30~0.60：
#       库内问题：0.239 / 0.286 / 0.289 / 0.290 / 0.292 / 0.297 / 0.310 /
#                 0.346 / 0.359 / 0.384 / 0.395 / 0.403 / 0.433 / 0.558
#       库外问题：0.453 / 0.483 / 0.525 / 0.527 / 0.558
#     总错误数（误拒 + 误答）：
#       0.41~0.43 → 2    0.44~0.45 → 1（最优）    0.46+ → 2（库外开始漏）
#     → 取 0.44，且 0.44~0.45 是稳定区间。
#
#   ⚠️ 仍剩下 1 条误拒（"Chroma 是干什么的" top1 = 0.558）—— 那是大模型笔记
#      对该概念覆盖不足导致的真实检索失败，**不是阈值能修的**。
#      要修只能补文档内容，或接 cross-encoder reranker 做精排判相关。
MAX_DISTANCE = _get_float("MAX_DISTANCE", 0.44)

TOP_K = _get_int("TOP_K", 3)              # 最终喂给模型的片段数
CANDIDATE_K = _get_int("CANDIDATE_K", 10)  # 每一路召回的候选数（融合前）

CHUNK_SIZE = _get_int("CHUNK_SIZE", 100)  # 每块多少字

# PDF 页眉页脚判定：某行出现的页数占比达到这个值才当页眉删掉。
# 0.5 的依据见 doc_loader.remove_repeated_lines 的注释（真页眉 99.7%，
# 章节标题/表格表头 24%~33%）。调低会误删正文，调高会漏删页眉。
PDF_REPEAT_MIN_RATIO = _get_float("PDF_REPEAT_MIN_RATIO", 0.5)

# 建库时每批嵌入多少块。整本手册有几千块，一次性发给 Ollama 会超时
# （实测 2822 块的单请求直接 ReadTimeout）。64 是稳妥值。
EMBED_BATCH_SIZE = _get_int("EMBED_BATCH_SIZE", 64)

# 写入 Chroma 时每批多少块。Chroma 单次 add() 有硬上限（实测 5461 条），
# 超过会直接报错。2000 留出余量。
CHROMA_ADD_BATCH_SIZE = _get_int("CHROMA_ADD_BATCH_SIZE", 2000)

# ============================================================
# Agent
# ============================================================
MAX_STEPS = _get_int("MAX_STEPS", 4)      # ReAct 最多循环几步（防止无限调工具）

# ============================================================
# 存储路径
# ============================================================
KNOWLEDGE_DIR = _get("KNOWLEDGE_DIR", os.path.join(BASE_DIR, "knowledge"))
CHROMA_DIR = _get("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
DB_PATH = _get("DB_PATH", os.path.join(BASE_DIR, "rag_meta.db"))
COLLECTION_NAME = _get("COLLECTION_NAME", "knowledge")

# ============================================================
# 接口服务
# ============================================================
API_HOST = _get("API_HOST", "127.0.0.1")
API_PORT = _get_int("API_PORT", 8000)


def describe():
    """返回当前生效的配置（隐藏路径细节，只给排查问题时看）

    启动时打一行，能立刻看出"你连的是哪个 Ollama / 用的哪个模型"，
    避免排查半天发现是环境变量没生效。
    """
    return (
        f"模型服务={OLLAMA_BASE} | 生成模型={CHAT_MODEL} | 嵌入模型={EMBED_MODEL} | "
        f"阈值={MAX_DISTANCE} | top_k={TOP_K} | 块大小={CHUNK_SIZE} | "
        f"Agent步数={MAX_STEPS} | 超时={REQUEST_TIMEOUT}s"
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("当前生效配置：")
    for key in (
        "OLLAMA_BASE", "OLLAMA_URL", "EMBED_URL", "CHAT_MODEL", "EMBED_MODEL",
        "REQUEST_TIMEOUT", "MAX_DISTANCE", "TOP_K", "CANDIDATE_K", "CHUNK_SIZE",
        "MAX_STEPS", "KNOWLEDGE_DIR", "CHROMA_DIR", "DB_PATH", "COLLECTION_NAME",
        "API_HOST", "API_PORT",
    ):
        print(f"  {key:<16} = {globals()[key]}")
    print("\n" + describe())
