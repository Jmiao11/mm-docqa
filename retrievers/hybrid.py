"""
retrievers/hybrid.py —— dense(语义) + BM25(字面) 双路召回，RRF 融合。

互补：dense 懂语义、对精确词/编号/专名不敏感；BM25 懂字面、不懂语义。
融合用 RRF（只用名次不用分数）：score = Σ 1/(k_rrf + rank)，免去 dense 的 cosine
与 BM25 原始分量纲不可比的标定问题。k_rrf=60 为原论文经验值。

真相源：Chroma（dense 的存储）持久化、累积；BM25 是内存索引、重启即空。
故让 Chroma 当唯一真相源，BM25 从 collection.get() 全量派生、随时重建 —— 两路
语料永远一致、重启自愈、幂等。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import jieba

from core.interfaces import Chunk, Retrieved, Retriever


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


class _BM25:
    """最小 BM25(Okapi)：TF 饱和(k1) + 文档长度归一(b) + IDF。
    self.k1 = 1.5   # 控制词频饱和速度（词出现多少次算"够了"） 词频到几次算"饱和"，越大饱和越慢
    self.b  = 0.75  # 控制文档长度的惩罚力度
    self.docs       # 所有文档的分词结果
    self.idf        # 每个词的稀有程度分数
    self.avgdl      # 所有文档的平均词数
    self.N          # 文档总数
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.N = 0

    def index(self, tokenized_docs: list[list[str]]) -> None:
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        df = Counter()
        for d in tokenized_docs:
            for w in set(d):
                df[w] += 1

        # 所有文档词数加起来 ÷ 文档数
        # 用来判断一篇文档算不算"长文档"
        self.avgdl = (sum(len(d) for d in tokenized_docs) / self.N) if self.N else 0.0

        # 算每个词的 IDF 逆文档频率（Inverse Document Frequency）
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, q_tokens: list[str]) -> list[float]:
        out = [0.0] * self.N
        if self.N == 0 or self.avgdl == 0:
            return out
        for i, d in enumerate(self.docs):
            dl = len(d)
            tf = Counter(d)
            s = 0.0
            for w in set(q_tokens):
                f = tf.get(w, 0)
                if f == 0:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


class HybridRetriever(Retriever):
    def __init__(self, dense, k_rrf: int = 60, candidate_k: int = 20):
        self.dense = dense
        self.k_rrf = k_rrf
        self.candidate_k = candidate_k
        self.bm25 = _BM25()
        self._chunks: list[Chunk] = []
        self._rebuild_bm25()          # 实例化即从 Chroma 派生 → 重启自愈

    def _rebuild_bm25(self) -> None:
        got = self.dense.collection.get()        # 全量
        ids = got.get("ids", []) or []
        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or [{}] * len(ids)
        # got长这样：
        # {
        #   "ids":       ["chunk_01", "chunk_02", ...],
        #   "documents": ["苹果很好吃。", "香蕉也不错。", ...],
        #   "metadatas": [{"source": "a.pdf", "start": 0}, ...]
        # }

        self._chunks = []
        for cid, text, md in zip(ids, docs, metas):
            md = dict(md or {})
            source = md.pop("source", "")
            start = md.pop("start", 0)
            end = md.pop("end", 0)

            self._chunks.append(Chunk(
                id=cid, text=text or "", source=source,
                start=start, end=end, metadata=md,    # 保留 image_path 等开放字段
            ))
        self.bm25.index([_tokenize(c.text) for c in self._chunks])
        # _tokenize把每段文本分词
        # ["苹果很好吃。"] → ["苹果", "很", "好吃"]
        # 列表推导式把所有 chunk 的文本都分词，得到一个二维列表（每个元素是一个词列表），整体喂给 bm25.index()，
        # BM25 内部计算每个词在每个文档里的词频和逆文档频率，建好索引。

    def index(self, chunks: list[Chunk]) -> None:
        self.dense.index(chunks)      # 先写 Chroma（唯一真相源）
        self._rebuild_bm25()          # 再从 Chroma 全量重建 BM25 → 两路一致

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        dense_hits = self.dense.retrieve(query, self.candidate_k)
        # dense返回语义最近的20个chunk

        bm25_scores = self.bm25.scores(_tokenize(query))
        # bm25给所有chunk打分，返回一个分数列表
        # [0.0, 2.3, 0.0, 1.1, 0.0, ...]

        bm25_order = sorted(range(len(self._chunks)),
                            key=lambda i: bm25_scores[i], reverse=True)[:self.candidate_k]
        # 按分数排序，取前20名的下标
        # [1, 3, 7, ...]  ← 这是在self._chunks里的位置下标

        rrf: dict[str, float] = defaultdict(float)
        # defaultdict(float) 是 collections 模块，当你试图访问字典里一个根本不存在的键（Key）时，它绝对不会像普通字典那样崩溃报错（KeyError），而是会自动帮你用默认值补齐。
        pool: dict[str, Chunk] = {}

        for rank, r in enumerate(dense_hits, start=1):
            rrf[r.chunk.id] += 1.0 / (self.k_rrf + rank)
            pool[r.chunk.id] = r.chunk

        for rank, idx in enumerate(bm25_order, start=1):
            if bm25_scores[idx] <= 0:
                continue  # 分数为0说明完全没有关键词命中，跳过
            c = self._chunks[idx]
            rrf[c.id] += 1.0 / (self.k_rrf + rank)
            pool.setdefault(c.id, c)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Retrieved(chunk=pool[cid], score=score) for cid, score in fused]