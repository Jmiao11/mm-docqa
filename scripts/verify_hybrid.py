"""3.2 验收：真实 PDF 上对比 dense vs hybrid 检索。运行：python verify_hybrid.py"""
import glob
from core.config import get_embedder
from core.paths import CHROMA_DIR
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from core.paths import CHROMA_DIR, DATA_DIR          # ← 直接导入


pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
doc = parse_pdf(pdf)
chunks = SemanticChunker(max_size=200, overlap_sentences=1).split(doc)
print(f"PDF: {doc.source} | semantic 切 {len(chunks)} 块\n")

emb = get_embedder("BAAI/bge-small-zh-v1.5")
dense = DenseRetriever(embedder=emb, persist_dir=str(CHROMA_DIR), collection_name="verify_hybrid")
dense.collection.delete(where={"source": {"$ne": ""}})   # 清空旧向量
dense.index(chunks)
hybrid = HybridRetriever(dense)        # 从同一集合派生 BM25

# 挑"精确词/编号"类问题：dense 易漏、BM25 易中
for q in ["主题数最终确定为几个", "困惑度最低对应的主题数", "供给型政策工具"]:
    d = [r.chunk.id[:8] for r in dense.retrieve(q, 5)]
    h = [r.chunk.id[:8] for r in hybrid.retrieve(q, 5)]
    print(f"Q: {q}")
    print("  dense  top5:", d)
    print("  hybrid top5:", h)
    top = hybrid.retrieve(q, 1)[0].chunk.text.strip().replace("\n", " ")[:60]
    print("  hybrid top1 文本:", top, "…\n")