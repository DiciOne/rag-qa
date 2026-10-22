# -*- coding: utf-8 -*-
"""
个人知识库问答系统（项目 1）
============================
功能：把 knowledge/ 里的文档存入 Chroma，支持自然语言问答

运行：
  1. 首次运行（建库）：  python rag.py
  2. 之后问答：          python rag.py
     （程序会自动检测：库不存在就重建，存在就直接问答）

技术栈：Python + Chroma（向量库）+ Ollama（bge-m3 嵌入 + qwen2.5:3b 回答）
"""

import os
import sys
import glob
import logging
import time
import chromadb
import numpy as np
from chromadb.utils import embedding_functions

import config
import doc_loader          # 文档加载与清洗模块（支持 txt / md / pdf）
from doc_store import DocStore                    # SQLite 业务元数据 + 问答历史
from qa_core import QAEngine, ModelServiceError   # 公共问答核心（混合检索 + 生成）

# ---------- 配置（全部来自 config.py，支持 .env 覆盖）----------
KNOWLEDGE_DIR = config.KNOWLEDGE_DIR   # 文档目录
CHROMA_DIR = config.CHROMA_DIR         # 向量库存储位置
CHUNK_SIZE = config.CHUNK_SIZE         # 每块多少字
COLLECTION_NAME = config.COLLECTION_NAME
EMBED_MODEL = config.EMBED_MODEL       # 嵌入模型

logger = logging.getLogger(__name__)


# ============================================================
# 模块 1：读取文档
# 需求：写 load_documents()，读取 KNOWLEDGE_DIR 下所有 .txt/.md 文件，
#       返回 [(文件名, 全文内容), ...]
# ============================================================
def load_documents():
    """读取 knowledge/ 下所有支持的文档（txt / md / pdf）

    返回 [(文件名, 文本, 元信息), ...]——元信息里带 type / pages，
    后续写入 SQLite 做文档管理时要用。
    """
    documents = []
    all_files = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.*")))
    for fp in all_files:
        if not fp.lower().endswith(doc_loader.supported_extensions()):
            continue
        name, text, meta = doc_loader.load_document(fp)
        print(f"  📄 {name}（{meta['type']}，{meta['pages']} 页，{len(text)} 字符）")
        if len(text.strip()) < 10:
            print("     ⚠️  内容过少，跳过")
            continue
        documents.append((name, text, meta))
    return documents


# ============================================================
# 模块 2：文档切块
# ============================================================
def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0 , len(text) , chunk_size)]


# ============================================================
# 模块 3：建库（向量化 + 存入 Chroma）
# 需求：写 build_collection(documents)：
#   1. 创建 embedding 函数（bge-m3）
#   2. 创建持久化 client：chromadb.PersistentClient(path=CHROMA_DIR)
#   3. 删除旧 collection（如果存在），再新建
#   4. 遍历每个文档：切块后 collection.add 存入
#      - 每块的 id 用 "doc序号_块序号"
#      - 每块附带 metadata={"source": 文件名}（以后能追溯来源）
#   5. 打印存入多少块
# ============================================================
def build_collection(documents):
    """建库：切块 → 分批嵌入 → 写入 Chroma

    ★ 两个踩过的坑（都是在真实手册上炸出来的）
    ------------------------------------------------------------
    坑 1：一次性把全部块交给 collection.add()
        少量文档时没问题。但上了 329 页的西门子手册（2822 块），
        chroma 会把全部文本塞进**一个** embedding 请求，
        单请求耗时 6 分钟以上，直接 httpcore.ReadTimeout 断掉。
        → 修法：分批嵌入（EMBED_BATCH_SIZE），每批几十条，失败可重试。

    坑 2：先 delete_collection 再建新的（非原子）
        嵌入失败时，旧库已经被删光了 —— 数据丢失，而错误信息里
        只写"超时"，完全看不出库已经空了。
        → 修法：**先把 embedding 全部算完，再删旧库**。算的过程中
          出错，旧库原样保留；算完了才动库，此时只是纯写入，很快。
    """
    ef = embedding_functions.OllamaEmbeddingFunction(
        url=config.EMBED_URL,
        model_name=EMBED_MODEL,
    )

    # ---------- 1. 切块 ----------
    all_chunks, all_ids, all_metas = [], [], []
    for doc_idx, (filename, content, _meta) in enumerate(documents):
        chunks = chunk_text(content)
        print(f"   ✂️  {filename} → {len(chunks)} 块")
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc_{doc_idx}_{chunk_idx}")
            all_metas.append({"source": filename})

    total = len(all_chunks)
    if total == 0:
        print("⚠️  没有可入库的内容")
        return None

    # ---------- 2. 算 embedding（优先读缓存，失败时旧库不受影响）----------
    batch_size = config.EMBED_BATCH_SIZE
    all_embeddings = _load_embedding_cache(all_ids, total)
    if all_embeddings is not None:
        print(f"   ⚡ 命中 embedding 缓存，跳过计算（{total} 块）")
    else:
        print(f"   🔢 共 {total} 块，分批嵌入（每批 {batch_size} 条）...")
        all_embeddings = []
        started = time.time()
        for start in range(0, total, batch_size):
            batch = all_chunks[start:start + batch_size]
            all_embeddings.extend(ef(batch))
            done = min(start + batch_size, total)
            pct = done * 100 / total
            elapsed = time.time() - started
            eta = elapsed / done * (total - done) if done else 0
            print(f"\r      进度 {done}/{total} ({pct:5.1f}%)  已用 {elapsed:5.0f}s  "
                  f"预计剩余 {eta:4.0f}s", end="", flush=True)
        print()
        _save_embedding_cache(all_ids, all_embeddings)

    # ---------- 3. 此时才删旧库并写入（纯写入，不会再碰模型）----------
    #
    # ★ 坑 3：Chroma 单次 add() 有批量上限（实测 5461 条）。
    #   一口气写 11048 块会报 "Batch size ... is greater than max batch size"，
    #   而旧库这时已经被删掉了 —— 又丢一次数据。
    #   所以写入也要分批（CHROMA_ADD_BATCH_SIZE，默认 2000）。
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # 显式设 HNSW 参数：默认的 search_ef 太小，11k 规模上会把真近邻漏掉
    # （实测必须把 n_results 提到 1000 才找得到，说明是搜索广度不够）。
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=ef,
        metadata={
            "hnsw:space": config.HNSW_SPACE,
            "hnsw:search_ef": config.HNSW_SEARCH_EF,
            "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
            "hnsw:M": config.HNSW_M,
        },
    )

    add_batch = config.CHROMA_ADD_BATCH_SIZE
    print(f"   💾 分批写入向量库（每批 {add_batch} 块）...")
    for start in range(0, total, add_batch):
        end = min(start + add_batch, total)
        collection.add(
            documents=all_chunks[start:end],
            embeddings=all_embeddings[start:end],
            ids=all_ids[start:end],
            metadatas=all_metas[start:end],
        )
        print(f"\r      已写入 {end}/{total}", end="", flush=True)
    print()

    got = collection.count()
    if got != total:
        raise RuntimeError(f"写入不完整：期望 {total} 块，实际 {got} 块")
    print(f"✅ 建库完成，共存入 {got} 块")
    return collection


# ============================================================
# embedding 缓存
# 算 11048 块的向量要 32 分钟。调试切块/清洗时反复重建，每次都重算是
# 纯浪费。所以按 chunk id 做缓存：id 全一样就直接复用。
# 缓存文件在 .gitignore 里（约 45 MB，不该进仓库）。
# ============================================================
def _cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "embeddings_cache.npz")


def _load_embedding_cache(all_ids, total):
    """命中则返回 embeddings 列表，否则 None"""
    path = _cache_path()
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        cached_ids = list(data["ids"])
        if cached_ids == list(all_ids):
            emb = data["embeddings"]
            if len(emb) == total:
                print(f"   （缓存文件 {os.path.getsize(path) / 1024 / 1024:.0f} MB，"
                      f"来自上次构建）")
                return [row.tolist() for row in emb]
        print("   （缓存已过期：切块结果变了，重新计算）")
    except Exception as exc:
        print(f"   （缓存读取失败，重新计算：{exc}）")
    return None


def _save_embedding_cache(all_ids, all_embeddings):
    path = _cache_path()
    try:
        np.savez_compressed(
            path,
            ids=np.array(all_ids, dtype=object),
            embeddings=np.array(all_embeddings, dtype=np.float32),
        )
        print(f"   💾 embedding 已缓存：{os.path.getsize(path) / 1024 / 1024:.0f} MB")
    except Exception as exc:
        print(f"   ⚠️ 缓存写入失败（不影响建库）：{exc}")


def save_metadata(documents):
    """把文档元数据与块同步写入 SQLite（供文档管理 / Agent 的工具使用）

    两个存储各司其职：
      向量库（Chroma）→ 语义检索
      SQLite          → 业务元数据（文档列表、块数、统计）
    """
    store = DocStore()
    store.reset()                      # 重建时先清空，避免残留旧文档
    for name, text, meta in documents:
        doc_id = store.add_document(name, meta.get("type", ""), meta.get("pages", 1), len(text))
        store.add_chunks(doc_id, name, chunk_text(text))
    s = store.stats()
    print(f"✅ 元数据已写入 SQLite：{s['doc_count']} 个文档 / {s['chunk_count']} 个块")
    store.close()


# ============================================================
# 模块 4：问答
# 检索与生成已抽取到 qa_core.QAEngine（命令行 / 接口 / 调试共用同一实现），
# 这里只做转发：engine.answer(question) -> (回答, 来源列表)
# ============================================================
def query(question, engine):
    """提问并返回 (回答, 来源列表)。检索与生成逻辑在 qa_core.QAEngine 中。"""
    result = engine.answer(question)
    return result["answer"], result["sources"]

# ============================================================
# 主程序
# ============================================================
def main():
    # 日志：一行一条，用于事后排查（问了什么、检索多久、总计多久）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("生效配置：%s", config.describe())

    # 检查库是否已存在（--rebuild 强制重建）
    force_rebuild = "--rebuild" in sys.argv
    db_exists = (not force_rebuild) and os.path.exists(CHROMA_DIR) and os.path.exists(
        os.path.join(CHROMA_DIR, "chroma.sqlite3"))

    ef = embedding_functions.OllamaEmbeddingFunction(
        url=config.EMBED_URL,
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if db_exists:
        print("📚 检测到已有知识库，直接加载...")
        collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    else:
        print("📚 正在建库...")
        documents = load_documents()
        print(f"   读取到 {len(documents)} 个文档")
        collection = build_collection(documents)
        save_metadata(documents)        # 同步写入 SQLite 元数据

    # SQLite 连接全进程共用一份：业务元数据 + 问答历史都写这里
    store = DocStore()
    engine = QAEngine(collection, store=store)   # 混合检索 + 生成 + 记录问答历史

    # ---- 是否启用 Agent 模式（--agent）----
    use_agent = "--agent" in sys.argv
    agent = None
    if use_agent:
        from agent_core import Agent

        agent = Agent(engine, store)
        print("✅ 知识库就绪（Agent 模式：模型可调用工具）")
        print(f"   可用工具：{', '.join(agent.tools)}")
    else:
        print("✅ 知识库就绪（混合检索模式）")
    print("   输入 exit 退出\n")

    while True:
        q = input("问题: ").strip()
        if q in ("exit", "退出", "quit"):
            break
        if not q:
            continue

        try:
            if agent is not None:
                print(f"助手: {agent.run(q)}\n")
            else:
                answer, sources = query(q, engine)
                print(f"回答: {answer}\n")
                print(f"来源:{','.join(sources)}\n")
        except ModelServiceError as e:
            # 模型服务挂了不能让整个程序退出，提示后继续等下一个问题
            print(f"⚠️  {e}\n")
            continue


if __name__ == "__main__":
    main()
