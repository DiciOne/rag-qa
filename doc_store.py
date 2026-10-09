# -*- coding: utf-8 -*-
"""
文档元数据管理（SQLite）
========================
职责：管理知识库的"业务元数据"——哪些文档入库了、每个文档切了多少块、什么时候入库的。

为什么单独用 SQL 管这些？
    向量库（Chroma）擅长"语义检索"，但不适合做业务管理（列出文档、统计、事务）。
    企业里的常见架构是：**向量库存语义 + 关系库存业务元数据**，两边用文档名/ID 关联。

跑法（自测）：python doc_store.py
"""

import datetime
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_meta.db")


class DocStore:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row      # 让查询结果能按列名取值
        self._init_schema()

    def _init_schema(self):
        """建表（已存在就跳过，所以可以重复运行）"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                type       TEXT,
                pages      INTEGER,
                chars      INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id   INTEGER NOT NULL,
                chunk_no INTEGER,
                source   TEXT,
                content  TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            );
        """)
        self.conn.commit()

    # ============================================================
    # 写入部分（已写好，供你参考 INSERT 的写法）
    # ============================================================
    def add_document(self, name, doc_type, pages, chars):
        """新增文档记录，返回 doc_id（同名文档会先删掉旧的，重新插入）"""
        self.conn.execute("DELETE FROM documents WHERE name = ?", (name,))
        cur = self.conn.execute(
            "INSERT INTO documents (name, type, pages, chars, created_at) VALUES (?,?,?,?,?)",
            (name, doc_type, pages, chars,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_chunks(self, doc_id, source, chunks):
        """批量写入该文档的所有块，返回写入条数"""
        rows = [(doc_id, i, source, c) for i, c in enumerate(chunks)]
        self.conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_no, source, content) VALUES (?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    # ============================================================
    # 查询部分（★ 你来写 ★）
    # ============================================================
    def list_documents(self):
        """列出所有文档：名字、类型、块数、字符数、入库时间（按块数从多到少排序）

        返回 list[dict]，每个 dict 形如：
            {"name": "a.txt", "type": "txt", "chunks": 5, "chars": 442, "created_at": "..."}

        ★ 提示：documents LEFT JOIN chunks + GROUP BY + COUNT(c.id)
              —— 注意别用 COUNT(*)，会数错（想想为什么）
        """
        # TODO: 你的 SQL 写在这里 
        sql = """SELECT d.name , d.type , d.chars , d.created_at,COUNT(c.id) AS chunks
        FROM documents d
        LEFT JOIN chunks c ON c.doc_id=d.id
        GROUP BY d.id
        ORDER BY chunks DESC"""
        rows = self.conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def stats(self):
        """总览统计，返回 {"doc_count": 文档数, "chunk_count": 块数, "total_chars": 字符总数}

        ★ 提示：需要三条聚合查询（或者一条也行，看你）
        """
        # TODO: 你的 SQL 写在这里
        doc_count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(id) FROM chunks").fetchone()[0]
        total_chars = self.conn.execute("SELECT SUM(chars) FROM documents").fetchone()[0]
        return {"doc_count":doc_count , "chunk_count":chunk_count,"total_chars":total_chars}
    

    def delete_document(self, name):
        """删除一个文档及其所有块，返回 (删除的文档行数, 删除的块行数)

        ★ 提示：先按 doc_id 删块，再删文档本身（否则不知道 doc_id 了）
              每条 DELETE 执行完可以用 cur.rowcount 拿到影响行数
        """
        # TODO: 你的 SQL 写在这里
        row = self.conn.execute("SELECT id FROM documents WHERE name = ?",(name,)).fetchone()
        if row is None:
            return (0,0)
        doc_id = row["id"]

        cur = self.conn.execute("DELETE FROM chunks WHERE doc_id = ?",(doc_id,))
        chunk_rows = cur.rowcount

        cur = self.conn.execute("DELETE FROM documents WHERE id=?",(doc_id,))
        doc_rows = cur.rowcount

        self.conn.commit()
        return (doc_rows , chunk_rows)
    
    def close(self):
        self.conn.close()


# ============================================================
# 自测：写完上面三个函数后运行 python doc_store.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    store = DocStore("test_meta.db")

    # 造点测试数据
    store.add_document("手册.pdf", "pdf", 2, 258)
    store.add_document("笔记.txt", "txt", 1, 442)
    d_id = store.add_document("空文档.md", "md", 1, 0)      # 故意造一个没有块的文档
    store.add_chunks(1, "手册.pdf", ["块1", "块2", "块3"])
    store.add_chunks(2, "笔记.txt", ["块A", "块B"])

    print("=== list_documents() ===")
    for d in store.list_documents():
        print("  ", d)

    print("\n=== stats() ===")
    print("  ", store.stats())

    print("\n=== delete_document('笔记.txt') ===")
    print("  ", store.delete_document("笔记.txt"))

    print("\n=== 删除后再看 list_documents() ===")
    for d in store.list_documents():
        print("  ", d)

    store.close()
    print("\n（自测用的 test_meta.db 可以随时删掉）")
