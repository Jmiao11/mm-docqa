"""dense / hybrid / rerank 三者检索指标对比(k=3,5)+ 延迟。python scripts/eval_rerank.py"""
import glob, time
from core.config import get_embedder, get_reranker
from core.paths import CHROMA_DIR, DATA_DIR
from core.interfaces import RAGResult, EvalSample
from ingest.parser import parse_pdf
from chunkers.semantic import SemanticChunker
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.rerank import RerankRetriever
from evaluators.retrieval import RetrievalEvaluator

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
dense.collection.delete(where={"source": {"$ne": ""}})
dense.index(chunks)
hybrid = HybridRetriever(dense)
rerank = RerankRetriever(base=hybrid, reranker=get_reranker("BAAI/bge-reranker-base"), candidate_k=20)

retrievers = [("dense", dense), ("hybrid", hybrid), ("rerank", rerank)]
for k in (3, 5):
    ev = RetrievalEvaluator(k=k)
    print(f"\n=== top-k={k} ===")
    for name, retr in retrievers:
        t = time.perf_counter()
        results = [RAGResult(query=q, retrieved=retr.retrieve(q, k), answer="") for q, _ in SAMPLES]
        ms = (time.perf_counter() - t) / len(SAMPLES) * 1000
        print(f"[{name:7}] {ev.evaluate(results, samples)}  ~{ms:.0f}ms/q")