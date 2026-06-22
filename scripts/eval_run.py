"""对比 dense vs hybrid 的检索指标。python scripts/eval_run.py

黄金集已外置到 evaluators/golden.jsonl（单一真相源），用 relevant_chunk_ids 字段。
"""
import glob
from core.config import get_embedder
from core.paths import CHROMA_DIR, DATA_DIR
from core.interfaces import RAGResult
from evaluators.golden import load_golden
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from evaluators.retrieval import RetrievalEvaluator

# 黄金评估集：从 evaluators/golden.jsonl 读，检索侧只用 relevant_chunk_ids 非空的题
samples = [s for s in load_golden() if s.relevant_chunk_ids]

pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
chunks = SemanticChunker(max_size=200, overlap_sentences=1).split(parse_pdf(pdf))

emb = get_embedder("BAAI/bge-small-zh-v1.5")

# eval_set 是纯派生集合(脚本每次全量重建)。用「删集合再重建」替代「delete(where=全删)」：
# 后者会路由到 chroma compactor 去读 HNSW 段，跨进程残留的半落盘段会让它崩
# (InternalError: Error loading hnsw index)。删整集合不读旧段，干净幂等；
# DenseRetriever 构造时以 cosine 重新 get_or_create，不丢距离度量配置。
import chromadb
try:
    chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection("eval_set")
except Exception:
    pass
dense = DenseRetriever(embedder=emb, persist_dir=str(CHROMA_DIR), collection_name="eval_set")
dense.index(chunks)

hybrid = HybridRetriever(dense)        # 从同一集合派生 BM25

ev = RetrievalEvaluator(k=5)
print(f"评估集 {len(samples)} 题 | 切块 {len(chunks)} | top-k=5\n")
for name, retr in [("dense", dense), ("hybrid", hybrid)]:
    results = [RAGResult(query=s.query, retrieved=retr.retrieve(s.query, 5), answer="") for s in samples]
    print(f"[{name:7}]", ev.evaluate(results, samples))


# 末尾追加：看每题每个检索器把第一个相关块排在第几名
print("\n--- per-query 第一个相关块的排名(1-5，-表示top5外) ---")
for s in samples:
    relevant = set(s.relevant_chunk_ids)
    row = []
    for name, retr in [("dense", dense), ("hybrid", hybrid)]:
        got = [r.chunk.id for r in retr.retrieve(s.query, 5)]
        rank = next((i + 1 for i, c in enumerate(got) if c in relevant), "-")
        row.append(f"{name}:{rank}")
    print(f"  {s.query[:18]:20} {'  '.join(row)}")