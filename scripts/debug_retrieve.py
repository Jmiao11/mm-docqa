"""调试：看真实 PDF 入库后，检索到底召回了哪些块。"""
from core.config import RAGConfig, build_pipeline

cfg = RAGConfig(retriever="dense", generator="llm", collection_name="docqa")
pipeline = build_pipeline(cfg)
retriever = pipeline.retriever

# 先看库里到底有多少块
print("Chroma 集合里的块数:", retriever.collection.count())

for q in ["这篇论文用了什么研究方法", "LDA主题模型", "数字经济政策"]:
    print("\n" + "=" * 60)
    print("问:", q)
    hits = retriever.retrieve(q, k=3)
    print(f"召回 {len(hits)} 块:")
    for i, r in enumerate(hits, 1):
        text = r.chunk.text.strip().replace("\n", " ")[:120]
        print(f"  [{i}] score={r.score:.3f} source={r.chunk.source}")
        print(f"      {text}…")