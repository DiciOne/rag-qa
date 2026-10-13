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
import chromadb
from chromadb.utils import embedding_functions

import doc_loader          # 文档加载与清洗模块（支持 txt / md / pdf）
from qa_core import QAEngine, ModelServiceError   # 公共问答核心（混合检索 + 生成）

# ---------- 配置 ----------
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")  # 文档目录
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")      # 向量库存储位置
CHUNK_SIZE = 100        # 每块多少字
COLLECTION_NAME = "knowledge"
EMBED_MODEL = "bge-m3"  # 嵌入模型


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
    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # 删除旧的（如果存在）
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    all_chunks = []
    all_ids = []
    all_metas = []
    for doc_idx , (filename , content , _meta) in enumerate(documents):
        chunks = chunk_text(content)
        for chunk_idx , chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc_{doc_idx}_{chunk_idx}")
            all_metas.append({"source":filename})
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metas
    )

    print(f"✅ 建库完成，共存入 {collection.count()} 块")
    return collection


def save_metadata(documents):
    """把文档元数据与块同步写入 SQLite（供文档管理 / Agent 的工具使用）

    两个存储各司其职：
      向量库（Chroma）→ 语义检索
      SQLite          → 业务元数据（文档列表、块数、统计）
    """
    from doc_store import DocStore

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
    # 检查库是否已存在（--rebuild 强制重建）
    force_rebuild = "--rebuild" in sys.argv
    db_exists = (not force_rebuild) and os.path.exists(CHROMA_DIR) and os.path.exists(
        os.path.join(CHROMA_DIR, "chroma.sqlite3"))

    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
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

    engine = QAEngine(collection)      # 混合检索 + 生成，统一由 qa_core 提供

    # ---- 是否启用 Agent 模式（--agent）----
    use_agent = "--agent" in sys.argv
    agent = None
    if use_agent:
        from agent_core import Agent
        from doc_store import DocStore

        agent = Agent(engine, DocStore())
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
