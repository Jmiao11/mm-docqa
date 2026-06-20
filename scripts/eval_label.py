"""列出每个查询的候选块(带完整 id)，供人工标注相关块。python scripts/eval_label.py

题目从 data/golden.jsonl 读（只需 query）。标完把相关块 id 填回 golden.jsonl 的
relevant_chunk_ids 字段，扩集 = 在 golden.jsonl 追加一行后重跑本脚本拿候选。
"""
import glob
from core.config import get_embedder
from core.paths import CHROMA_DIR, DATA_DIR
from evaluators.golden import load_queries
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever

# 题目来自单一真相源 data/golden.jsonl
QUERIES = load_queries()

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