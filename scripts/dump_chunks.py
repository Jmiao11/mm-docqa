"""诊断：dump 某文档的全部块原文，看表格结构在解析/切块后是否还在。"""
import chromadb
from core.paths import CHROMA_DIR

col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("docqa_v2")
res = col.get(where={"source": "研究生选课.pdf"},
              include=["documents", "metadatas"])

print("块数:", len(res["documents"]))
for i, (doc, md) in enumerate(zip(res["documents"], res["metadatas"])):
    print(f"\n--- chunk {i}  kind={md.get('kind')}  page={md.get('page')}  "
          f"start={md.get('start')} end={md.get('end')} ---")
    print(doc)