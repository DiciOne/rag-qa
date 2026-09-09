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
import glob
import chromadb
from chromadb.utils import embedding_functions

# ---------- 配置 ----------
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")  # 文档目录
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")      # 向量库存储位置
CHUNK_SIZE = 100        # 每块多少字
COLLECTION_NAME = "knowledge"
EMBED_MODEL = "bge-m3"  # 嵌入模型
CHAT_MODEL = "qwen2.5:3b"

# ---------- 工具：调用大模型 ----------
import requests

def ask(prompt, temperature=0.3):
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    })
    return resp.json()["message"]["content"]


# ============================================================
# 模块 1：读取文档
# 需求：写 load_documents()，读取 KNOWLEDGE_DIR 下所有 .txt/.md 文件，
#       返回 [(文件名, 全文内容), ...]
# 提示：glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt")) 找文件
#       open(file, encoding="utf-8").read() 读内容
# ============================================================
def load_documents():
    documents = []
    # 你的代码写在这里 ↓
    all_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.*"))
    for fp in all_files:
        if fp.endswith((".txt", ".md")):
            with open(fp, "r" , encoding="utf-8") as f:
                text = f.read()
                name = os.path.basename(fp)
                documents.append((name , text))
    return documents


# ============================================================
# 模块 2：文档切块
# 需求：写 chunk_text(text, chunk_size)，按字数切块返回列表
#       （第二周写过，凭记忆写）
# ============================================================
def chunk_text(text, chunk_size=CHUNK_SIZE):
    # 你的代码写在这里 ↓
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
# 提示：add(documents=[...], ids=[...], metadatas=[...])
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

    # 你的代码写在这里 ↓
    all_chunks = []
    all_ids = []
    all_metas = []
    for doc_idx , (filename , content) in enumerate(documents):
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
    # 思路：遍历 documents，对每个 (filename, content) 切块，
    #       收集所有块、id、metadata，最后一次性 collection.add()
    # id 示例："0_0", "0_1", "1_0"（第0个文档第0块）
    # metadata 示例：{"source": "python笔记.txt"}

    print(f"✅ 建库完成，共存入 {collection.count()} 块")
    return collection


# ============================================================
# 模块 4：问答
# 需求：写 query(question, collection)：
#   1. collection.query(query_texts=[question], n_results=3) 检索
#   2. 把命中的块拼成"材料"（带上来源）
#   3. 拼提示词：基于材料回答，不知道就说不知道
#   4. ask() 返回回答
# ============================================================
def query(question, collection):
    # 你的代码写在这里 ↓
    results = collection.query(query_texts=[question], n_results=3)
    chunks = results["documents"][0]

    if not chunks:
        return "知识库中没有找到相关信息。"
    
    sources = results["metadatas"][0]   # 每个元素是 {"source": "文件名"} 字典
    # 正确做法：用 src["source"] 取出文件名，而不是把整个字典塞进去
    context = "\n".join(
        f"[来自{src['source']}]\n{chunk}" for chunk, src in zip(chunks, sources)
    )
    prompt = (
        "你是一个知识库问答助手。请只基于下面的材料回答问题，"
        "如果材料中没有相关信息，就说'知识库中没有找到相关信息'。\n"
        "材料：\n" + context + "\n问题：" + question
    )
    source_names_raw = [src['source'] for src in sources]
    source_names = list(set(source_names_raw))
    return ask(prompt) , source_names


# ============================================================
# 主程序
# ============================================================
def main():
    # 检查库是否已存在
    import shutil
    db_exists = os.path.exists(CHROMA_DIR) and os.path.exists(
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
        print("📚 首次运行，正在建库...")
        documents = load_documents()
        print(f"   读取到 {len(documents)} 个文档")
        collection = build_collection(documents)

    print("✅ 知识库就绪，开始问答（输入 exit 退出）\n")
    while True:
        q = input("问题: ").strip()
        if q in ("exit", "退出", "quit"):
            break
        if not q:
            continue
        answer , sources= query(q, collection)
        print(f"回答: {answer}\n")
        print(f"来源:{','.join(sources)}\n")


if __name__ == "__main__":
    main()
