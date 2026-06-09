"""对比 dense vs hybrid 的检索指标。python scripts/eval_run.py"""
import glob
from core.config import get_embedder
from core.paths import CHROMA_DIR, DATA_DIR
from core.interfaces import RAGResult, EvalSample
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from evaluators.retrieval import RetrievalEvaluator

# 黄金评估集：query -> 相关块 id（来自 eval_label 的人工标注）
SAMPLES = [
    ("本研究将政策工具分为哪三类", ["9b6fa194ac750cef", "501ae4da6dd4944f"]),
    ("最终确定的主题数是多少",     ["c6b0da78a3bbeed7", "0b3a3703936d20c8"]),
    ("困惑度曲线说明了什么趋势",   ["c6b0da78a3bbeed7", "3a00659286750425"]),
    ("本文采用了什么研究方法",     ["288b5395fa9c00c6", "3d184b31aeef046b"]),
    ("本文分析了多少份政策文本",   ["288b5395fa9c00c6"]),
    ("社会网络分析得出了什么结论", ["89b801eba3ae73b3", "4239fd1a5fbabfb4"]),
]
samples = [EvalSample(query=q, relevant_chunk_ids=ids) for q, ids in SAMPLES]

pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
chunks = SemanticChunker(max_size=200, overlap_sentences=1).split(parse_pdf(pdf))

emb = get_embedder("BAAI/bge-small-zh-v1.5")
dense = DenseRetriever(embedder=emb, persist_dir=str(CHROMA_DIR), collection_name="eval_set")
dense.collection.delete(where={"source": {"$ne": ""}})   # 清空旧向量，重新入库
dense.index(chunks)
hybrid = HybridRetriever(dense)        # 从同一集合派生 BM25

ev = RetrievalEvaluator(k=5)
print(f"评估集 {len(SAMPLES)} 题 | 切块 {len(chunks)} | top-k=5\n")
for name, retr in [("dense", dense), ("hybrid", hybrid)]:
    results = [RAGResult(query=q, retrieved=retr.retrieve(q, 5), answer="") for q, _ in SAMPLES]
    print(f"[{name:7}]", ev.evaluate(results, samples))


# 末尾追加：看每题每个检索器把第一个相关块排在第几名
print("\n--- per-query 第一个相关块的排名(1-5，-表示top5外) ---")
for q, ids in SAMPLES:
    relevant = set(ids)
    row = []
    for name, retr in [("dense", dense), ("hybrid", hybrid)]:
        got = [r.chunk.id for r in retr.retrieve(q, 5)]
        rank = next((i + 1 for i, c in enumerate(got) if c in relevant), "-")
        row.append(f"{name}:{rank}")
    print(f"  {q[:18]:20} {'  '.join(row)}")

