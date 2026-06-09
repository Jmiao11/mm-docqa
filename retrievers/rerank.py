"""
retrievers/rerank.py —— 两阶段检索的第二阶段：交叉编码重排。

bi-encoder(dense/BM25) 分别编码 query 和 doc 再比相似度：快(可预计算)但粗，
编码时 query 与 doc 互不可见。cross-encoder(reranker) 把 (query, doc) 拼成一条
输入过 transformer，全程交叉注意力 → 判相关性准得多；但每个候选一次前向、
无法预计算、不能全库跑。故：hybrid 先召回 top-N 候选 → cross-encoder 在 N 个里
重排 → top-k。对"正文+参考文献"混合块，cross-encoder 给引用部分打低分 → 压下去。
"""
from __future__ import annotations

from core.interfaces import Chunk, Retrieved, Retriever


class RerankRetriever(Retriever):
    def __init__(self, base, reranker=None,
                 model_name: str = "BAAI/bge-reranker-base", candidate_k: int = 20):
        self.base = base                  # 第一阶段：hybrid（或任意 Retriever）
        self.candidate_k = candidate_k
        if reranker is not None:
            self.model = reranker         # 注入：共享外部加载好的 CrossEncoder
        else:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)

    def index(self, chunks: list[Chunk]) -> None:
        self.base.index(chunks)           # 索引全交给第一阶段，自己不存

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        cands = self.base.retrieve(query, self.candidate_k)
        if not cands:
            return []
        scores = self.model.predict([(query, r.chunk.text) for r in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)
        return [Retrieved(chunk=r.chunk, score=float(s)) for r, s in ranked[:k]]