"""2.2 验证：SQLite 元数据库增删查 + 会话 + JSON往返。运行：python verify_db.py"""
from store.metadata_db import MetadataDB

db = MetadataDB("../data/app.db")   # 会在 data/ 下生成 app.db 文件

# 用你真实 PDF 的信息建一条记录
db.add_document("政策分析", "政策工具视域下的数字经济政策文本量化分析.pdf",
                n_pages=9, status="pending")
db.update_document("政策分析", n_chunks=42, status="indexed")
print("文档列表:")
for d in db.list_documents():
    print(f"  {d['id']} | {d['source']} | {d['n_pages']}页 | "
          f"{d['n_chunks']}块 | {d['status']}")

# 会话 + 带引用的消息
sid = "test_session"
db.add_message(sid, "user", "数字经济政策用了哪些政策工具？")
db.add_message(sid, "assistant", "主要包括供给型、需求型、环境型政策工具[1]。",
               sources=[{"n": 1, "id": "abc123", "source": "政策分析.pdf", "score": 0.71}])
print("\n会话记录:")
for m in db.get_messages(sid):
    print(f"  [{m['role']}] {m['content']}")
    if m["sources"]:
        print(f"        引用: {m['sources']}")

db.close()
print("\n✅ 跑通后，去 data/ 目录看看是不是多了个 app.db 文件")