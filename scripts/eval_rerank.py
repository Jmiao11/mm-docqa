"""dense / hybrid / rerank 三者检索指标对比(k=3,5)+ 延迟。python scripts/eval_rerank.py

题目从 evaluators/golden.jsonl 读(单一真相源)，只用有 relevant_chunk_ids 的题。
"""
import glob, time

from core.config import get_embedder, get_reranker
from core.paths import CHROMA_DIR, DATA_DIR
from core.interfaces import RAGResult
from evaluators.golden import load_golden
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.rerank import RerankRetriever
from evaluators.retrieval import RetrievalEvaluator

samples = [s for s in load_golden() if s.relevant_chunk_ids]

pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
chunks = SemanticChunker(max_size=200, overlap_sentences=1).split(parse_pdf(pdf))
emb = get_embedder("BAAI/bge-small-zh-v1.5")

# eval_set 是纯派生集合(每次全量重建)。用「删集合再重建」替代「delete(where=全删)」：
# 后者会路由到 chroma compactor 去读 HNSW 段，跨进程残留的半落盘段会让它崩。
# 删整集合不读旧段，干净幂等；DenseRetriever 构造时以 cosine 重新 get_or_create。
import chromadb
try:
    chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection("eval_set")
except Exception:
    pass
dense = DenseRetriever(embedder=emb, persist_dir=str(CHROMA_DIR), collection_name="eval_set")
dense.index(chunks)
hybrid = HybridRetriever(dense)
rerank = RerankRetriever(base=hybrid, reranker=get_reranker("BAAI/bge-reranker-base"), candidate_k=20)

retrievers = [("dense", dense), ("hybrid", hybrid), ("rerank", rerank)]
for k in (3, 5):
    ev = RetrievalEvaluator(k=k)
    print(f"\n=== top-k={k} | {len(samples)} 题 ===")
    for name, retr in retrievers:
        t = time.perf_counter()
        results = [RAGResult(query=s.query, retrieved=retr.retrieve(s.query, k), answer="")
                   for s in samples]
        ms = (time.perf_counter() - t) / len(samples) * 1000
        print(f"[{name:7}] {ev.evaluate(results, samples)}  ~{ms:.0f}ms/q")