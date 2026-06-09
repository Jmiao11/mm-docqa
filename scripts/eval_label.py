"""列出每个查询的候选块(带完整 id)，供人工标注相关块。python scripts/eval_label.py"""
import glob
from core.config import get_embedder
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from core.paths import CHROMA_DIR, DATA_DIR  # 清理了重复的导入

# 用我们刚才确定的“黄金评估集”替换掉原来的题目
QUERIES = [
    "本研究将政策工具分为哪三类",       # ← 新 Q1（答案明确）
    "最终确定的主题数是多少",
    "困惑度曲线说明了什么趋势",
    "本文采用了什么研究方法",
    "本文分析了多少份政策文本",         # ← 新 Q5（事实数据明确）
    "社会网络分析得出了什么结论",
]

# 坚不可摧的路径查找逻辑
pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
chunks = SemanticChunker(max_size=200, overlap_sentences=1).split(parse_pdf(pdf))

emb = get_embedder("BAAI/bge-small-zh-v1.5")
dense = DenseRetriever(embedder=emb, persist_dir=str(CHROMA_DIR), collection_name="eval_set")
dense.collection.delete(where={"source": {"$ne": ""}})
dense.index(chunks)

hy = HybridRetriever(dense)

for q in QUERIES:
    print("=" * 72); print("Q:", q)
    for r in hy.retrieve(q, 8):
        snip = r.chunk.text.strip().replace("\n", " ")[:46]
        print(f"  {r.chunk.id}  {snip}")