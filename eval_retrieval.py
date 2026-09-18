# -*- coding: utf-8 -*-
"""
检索效果评测（先度量，再优化）
================================
跑法：python eval_retrieval.py
前提：knowledge/ 里已有文档，且已建库（python rag.py --rebuild）

评测集 = 一批"问题 + 期望来源"的对照数据：
  - 有答案的问题：期望检索结果里出现指定来源文件
  - 无答案的问题：期望检索结果为空（不能瞎检索出内容）

指标：
  命中率 Hit@3    期望来源出现在前 3 条结果里的比例
  Top1 准确率     期望来源排在第一条的比例
  拒答率          无答案问题正确返回空的比例
"""

import chromadb
from chromadb.utils import embedding_functions

# ---------- 配置 ----------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge"
EMBED_MODEL = "bge-m3"
MAX_DISTANCE = 0.5      # 相似度阈值
K = 3                   # 取前 K 条
USE_HYBRID = True       # True = 混合检索（向量+BM25）；False = 纯向量（用于对比）

# ---------- 评测集 ----------
EVAL_SET = [
    # ---- 有答案的问题（source = 期望命中的文件）----
    {"q": "列表推导式怎么写", "source": "python笔记.txt"},
    {"q": "字典怎么取值才不会报错", "source": "python笔记.txt"},
    {"q": "函数的默认参数怎么用", "source": "python笔记.txt"},
    {"q": "类的方法第一个参数是什么", "source": "python笔记.txt"},
    {"q": "什么是提示词工程", "source": "大模型笔记.txt"},
    {"q": "RAG 是什么", "source": "大模型笔记.txt"},
    {"q": "temperature 参数怎么用", "source": "大模型笔记.txt"},
    {"q": "Chroma 是干什么的", "source": "大模型笔记.txt"},
    {"q": "messages 和 message 有什么区别", "source": "大模型笔记.txt"},
    {"q": "S7-1200 的工作电压是多少", "source": "S7-1200设备手册示例.pdf"},
    {"q": "数字量输入有多少点", "source": "S7-1200设备手册示例.pdf"},
    {"q": "IP 地址设置要注意什么", "source": "S7-1200设备手册示例.pdf"},
    {"q": "RUN 指示灯闪烁怎么处理", "source": "S7-1200设备手册示例.pdf"},
    {"q": "用什么软件进行设备组态", "source": "S7-1200设备手册示例.pdf"},
    # ---- 无答案的问题（source = None，期望检索结果为空）----
    {"q": "今天天气怎么样", "source": None},
    {"q": "公司食堂几点开门", "source": None},
    {"q": "怎么用 Excel 做数据透视表", "source": None},
]


# ============================================================
# 练习 1：检索函数
# 需求：写 retrieve(question, collection, k=K)，返回 [(块内容, 来源文件名, 距离), ...]
#   - 调 collection.query(query_texts=[question], n_results=k)
#   - 用 MAX_DISTANCE 过滤（距离 >= MAX_DISTANCE 的丢掉）
#   - 按距离升序排列
#   - 无结果返回 []（注意：类型要和有结果时一致，都是列表）
# 提示：结果在 results["documents"][0]、["metadatas"][0]、["distances"][0]
#       来源文件名在每项的 ["source"] 里
# ============================================================
def retrieve(question, collection, k=K):
    # 你的代码写在这里 ↓
    results = collection.query(query_texts=[question] , n_results=k)
    chunks = results['documents'][0]
    distances = results['distances'][0]

    relevant=[(chunk , src['source'] , dist) for chunk , src , dist in zip(chunks , results['metadatas'][0] , distances) if dist < MAX_DISTANCE]
    if not relevant :
        return []
    relevant=sorted(relevant , key=lambda x:x[2])

    return relevant
   

# ============================================================
# 练习 2：评测函数（核心）
# 需求：写 evaluate(collection)，返回一个字典：
#   {
#     "hit_rate":      命中率（保留 2 位小数）
#     "top1_rate":     Top1 准确率
#     "scope_rate":    拒答率（无答案问题返回空的比例）
#     "details":       [(问题, 期望来源, 实际来源列表, 是否命中), ...]
#   }
#
#   统计逻辑：
#     - 有答案的问题（source 不为 None）：
#         * 命中 = 期望来源出现在"实际来源列表"里
#         * Top1 = 实际结果的第 1 条的来源 == 期望来源
#     - 无答案的问题（source is None）：
#         * 正确 = 检索结果为空列表
#     - 比例 = 命中数 / 总数，用 round(x, 2)
#     - 注意除以 0 的情况（虽然评测集不为空，但养成习惯）
# ============================================================
def evaluate(search_fn):
    """search_fn(question) -> [(块内容, 来源, 分数), ...]，两种检索模式通用"""
    # 你的代码写在这里 ↓
    details=[]
    hit_count = 0
    top1_count = 0
    scope_correct = 0
    total_with_answer = 0
    total_without_answer = 0
    for i in EVAL_SET:
        is_hit = False
        relevant = search_fn(i['q'])
        # 统一契约：search_fn 返回 [(块内容, 来源文件名, 分数), ...]，来源已是字符串
        actual_sources = list(dict.fromkeys(src for _, src, _ in relevant))
        top1_source = relevant[0][1] if relevant else None
        if i['source'] is not None:
            total_with_answer +=1
            if i['source'] in actual_sources:
                hit_count +=1
                is_hit = True
            if  top1_source == i['source']:
                top1_count +=1
        else:
            total_without_answer += 1
            if len(actual_sources) == 0:
                scope_correct += 1
            is_hit = len(actual_sources) == 0
        details.append((i['q'],i['source'],actual_sources,is_hit))
    hit_rate = round(hit_count / total_with_answer , 2) if total_with_answer > 0 else 0.0
    top1_rate = round(top1_count / total_with_answer , 2) if total_with_answer > 0 else 0.0
    scope_rate = round(scope_correct / total_without_answer , 2) if total_without_answer >0 else 0.0
    return {
        "hit_rate":hit_rate,
        "top1_rate":top1_rate,
        "scope_rate":scope_rate,
        "details":details
    }

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings", model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    if USE_HYBRID:
        from hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(col, max_distance=MAX_DISTANCE)
        search_fn = lambda q: retriever.search(q, k=K)
        mode = "混合检索（向量 + BM25，RRF 融合）"
    else:
        search_fn = lambda q: retrieve(q, col, k=K)
        mode = "纯向量检索"

    print(f"检索模式：{mode}\n")
    result = evaluate(search_fn)

    print("=" * 55)
    print(f"命中率 Hit@{K}: {result['hit_rate']}")
    print(f"Top1 准确率:  {result['top1_rate']}")
    print(f"拒答率:       {result['scope_rate']}")
    print("=" * 55)
    print("\n明细（未命中的会标 ❌）：")
    for q, expected, actual, hit in result["details"]:
        mark = "✅" if hit else "❌"
        print(f"{mark} {q}")
        print(f"    期望: {expected}")
        print(f"    实际: {actual}")
