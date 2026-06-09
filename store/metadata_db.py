# store/metadata_db
"""
store/metadata_db.py —— SQLite 元数据库，管"应用状态"：上传了哪些文档、会话历史。
注意分工：这里只存元数据，不存向量（向量在 Chroma）。

对外暴露业务语义方法（add_document / list_documents / ...），
把 sqlite3 细节包在类里 —— 同一招接口隔离，API 层不碰 SQL。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,      -- 文档唯一标识（Document.id）
    source      TEXT NOT NULL,         -- 原始文件名
    n_pages     INTEGER DEFAULT 0,
    n_chunks    INTEGER DEFAULT 0,     -- 入库后回填
    status      TEXT DEFAULT 'pending',-- pending / indexed / failed
    uploaded_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,         -- user / assistant
    content     TEXT NOT NULL,
    sources     TEXT DEFAULT '',       -- 引用列表，JSON 字符串（单字段存不了列表）
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""
# ON messages(session_id)：指定目标。明确告诉数据库：“我要在 messages 这张表里，专门为 session_id 这一列建立目录。”

class MetadataDB:
    def __init__(self, db_path: str = "data/app.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 多线程下允许跨线程用同一连接
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row      # 让查询结果能按列名取，像 dict ：游标返回的 r 已经变成了一个极像字典的智能对象，它内部知道自己的列名！
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- documents ----------
    def add_document(self, doc_id: str, source: str, n_pages: int = 0,
                     n_chunks: int = 0, status: str = "pending") -> None:
        """新增或覆盖一条文档记录（同 id 覆盖，呼应确定性 id 的幂等思想）。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, source, n_pages, n_chunks, status, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, source, n_pages, n_chunks, status, time.time()),
        )
        self.conn.commit()

    # 这里的 fields 是 Python 的“关键字参数装包”语法
    def update_document(self, doc_id: str, **fields) -> None:
        """回填字段，如入库完成后更新 n_chunks 和 status。"""
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE documents SET {cols} WHERE id = ?",
            (*fields.values(), doc_id),
        )
        self.conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0          # True 表示确实删掉了一条


    # ---------- messages ----------
    def add_message(self, session_id: str, role: str, content: str,
                    sources: list[dict] | None = None) -> int:
        """记一条会话消息。sources 是结构化引用列表，序列化成 JSON 存。"""
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content,
             json.dumps(sources or [], ensure_ascii=False), time.time()),
        )
        # 必须使用 json.dumps() 把这个立体的 Python 数据结构，“压扁”成一串纯文本字符串：
        # '[{"id": "rag", "page": 12}, {"id": "vectordb", "page": 3}]'
        self.conn.commit()
        return cur.lastrowid


    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"]) if d["sources"] else []  # 还原成列表
            out.append(d)
        return out

    def close(self) -> None:
        self.conn.close()