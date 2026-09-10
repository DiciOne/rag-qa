# 📚 知识库问答系统（RAG）

基于 **RAG（检索增强生成）** 的知识库问答系统：把本地文档导入向量数据库，用自然语言提问，系统检索相关内容并生成回答，**支持来源追溯与相似度阈值过滤**。

提供**命令行**与 **HTTP 接口（FastAPI）** 两种使用方式，支持 Docker 容器化部署。

## ✨ 功能特点

- **本地私有化部署**：模型全部跑在本机（Ollama），数据不出本地，无 API 费用
- **语义检索**：向量检索（Chroma + bge-m3），能理解"意思相近但字面不同"的表达
- **相似度阈值过滤**：基于实测 distance 值设定阈值，过滤低相关结果，避免"检索到了但不相关"
- **防幻觉回答**：提示词约束仅基于检索材料回答，库外问题明确返回"未找到"，不编造
- **来源追溯**：每条回答标注信息来源文档，便于核查
- **HTTP 接口**：FastAPI 提供 GET/POST 问答接口，自动生成交互式 API 文档

## 🛠 技术栈

| 组件 | 用途 |
|---|---|
| Python | 主程序语言 |
| Chroma | 向量数据库（持久化存储） |
| Ollama | 本地模型运行环境 |
| bge-m3 | 嵌入模型（中文语义检索） |
| qwen2.5:3b | 问答生成模型 |
| FastAPI + Uvicorn | HTTP 接口服务 |
| Docker | 容器化部署 |

## 📦 环境要求

- Windows / Linux / macOS
- Python 3.10+
- [Ollama](https://ollama.com) 已安装并运行

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 拉取模型

```bash
ollama pull bge-m3      # 嵌入模型（中文检索效果好）
ollama pull qwen2.5:3b  # 问答模型
```

### 3. 放入知识文档

把 `.txt` / `.md` 文档放入 `knowledge/` 文件夹。

### 4a. 命令行方式

```bash
python rag.py
```

```
问题: 什么是RAG？
回答: RAG代表检索增强生成（Retrieval-Augmented Generation）...
来源: 大模型笔记.txt
```

### 4b. HTTP 接口方式

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

启动后访问：
- **接口文档**：http://localhost:8000/docs （FastAPI 自动生成，可在线测试）
- **GET 提问**：`http://localhost:8000/ask?question=什么是RAG`
- **POST 提问**：`POST /ask`，body 为 `{"question": "什么是RAG"}`

返回示例：

```json
{
  "question": "什么是RAG",
  "answer": "RAG代表检索增强生成（Retrieval-Augmented Generation）...",
  "sources": ["大模型笔记.txt"]
}
```

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t rag-qa .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db rag-qa
```

> **注意**：容器内的 `localhost` 指向容器自身，若 Ollama 运行在宿主机，需使用
> `--network host`（Linux）或在代码中将 `localhost` 改为宿主机 IP（Windows/Mac 用 `host.docker.internal`）。

## 🧠 工作原理

```
┌─────────┐  切块  ┌──────┐  向量化  ┌────────┐
│ 知识文档 │ ────→ │ 文档块 │ ──────→ │ Chroma │
└─────────┘        └──────┘          └────────┘
                                        ↑ 检索（带相似度阈值过滤）
用户问题 ──→ 向量化 ──→ 相关块 ──→ 拼成材料
                                        ↓
                                  qwen2.5:3b 生成回答
                                        ↓
                              {answer, sources} ← JSON 返回
```

## 💡 关键技术实践（踩坑记录）

### 1. 嵌入模型选型：nomic-embed-text → bge-m3

初始使用 `nomic-embed-text`，实测发现中文场景语义区分度不足——例如"上班时间灵活吗"无法正确命中"弹性工作制"，多个文档的 distance 值挤在一起（0.17~0.38）难以区分。换用 `bge-m3` 后检索命中率显著提升，distance 区分度明显改善。

### 2. 相似度阈值调优：解决"检索到但不相关"

**问题**：Chroma 的 `query(n_results=3)` 总会返回 3 个"最接近"的块，即使问题与知识库完全无关（如"公司食堂吃什么"），也会返回结果并标注来源，误导用户。

**诊断**：打印实际 distance 值对比——

| 问题 | distance | 相关性 |
|---|---|---|
| 什么是RAG（相关） | 0.39 | ✅ |
| 公司食堂吃什么（无关） | 0.59 | ❌ |

**解决**：取中间值设阈值 `distance < 0.5` 判定为相关，否则返回"未找到"。修复后无关问题 `sources` 正确返回空列表。

### 3. 防幻觉提示词设计

提示词中明确约束："请只基于下面的材料回答问题，如果材料中没有相关信息，就说'知识库中没有找到相关信息'"，并在检索结果为空时直接短路返回，不调用模型。

### 4. metadata 来源追溯

存入向量库时为每个块附带 `metadata={"source": 文件名}`，检索后取出该字段展示给用户，实现答案可溯源。

## 📁 目录结构

```
rag_qa/
├── rag.py             # 命令行版主程序（建库 + 问答）
├── api.py             # FastAPI 接口版
├── Dockerfile         # 容器化构建文件
├── requirements.txt   # 依赖清单
├── knowledge/         # 知识文档（txt/md）
├── chroma_db/         # 向量数据库（运行时生成，已 gitignore）
└── .gitignore
```

## 🔜 后续优化方向

- 支持 PDF / Word 文档解析（工业设备手册多为 PDF）
- 混合检索（关键词 + 向量）与重排序（rerank）提升召回质量
- 加入检索效果评测脚本，量化回答准确率
        
