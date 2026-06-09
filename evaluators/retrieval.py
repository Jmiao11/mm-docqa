"""
evaluators/retrieval.py —— 检索质量指标，实现 core.interfaces.Evaluator。

HitRate@k : 有几成查询，top-k 里至少命中 1 个相关块（"找没找到"）
Recall@k  : top-k 召回的相关块占全部相关块的比例（"找全没"）
MRR       : 第一个相关块排名的倒数的均值（"排得靠不靠前"），top-k 外算 0
results 与 samples 按 query 文本配对；只评 relevant_chunk_ids 非空的样本。
"""
from __future__ import annotations

from core.interfaces import Evaluator, RAGResult, EvalSample


class RetrievalEvaluator(Evaluator):
    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(self, results: list[RAGResult], samples: list[EvalSample]) -> dict[str, float]:
        by_query = {r.query: r for r in results}
        hit_sum = rec_sum = rr_sum = 0.0
        n = 0
        for s in samples:
            if not s.relevant_chunk_ids:
                continue

            r = by_query.get(s.query)
            if r is None:
                continue

            n += 1
            topk_ids = [ret.chunk.id for ret in r.retrieved[:self.k]]
            relevant = set(s.relevant_chunk_ids)

            hit_sum += 1.0 if any(cid in relevant for cid in topk_ids) else 0.0

            rec_sum += len(relevant & set(topk_ids)) / len(relevant)
            rr = 0.0
            for rank, cid in enumerate(topk_ids, start=1):
                if cid in relevant:
                    rr = 1.0 / rank
                    break
            rr_sum += rr
        if n == 0:
            return {f"HitRate@{self.k}": 0.0, f"Recall@{self.k}": 0.0, "MRR": 0.0, "n": 0.0}
        return {f"HitRate@{self.k}": round(hit_sum / n, 4),
                f"Recall@{self.k}": round(rec_sum / n, 4),
                "MRR": round(rr_sum / n, 4), "n": float(n)}