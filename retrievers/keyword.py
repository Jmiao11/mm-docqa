"""占位检索器：中文 bigram 词重叠打分。纯标准库，只为证明接口跑得通。
真正的语义检索在 Phase 1 的 retrievers/dense.py。"""
from __future__ import annotations

from collections import Counter

from core.interfaces import Chunk, Retrieved, Retriever

def _bigrams(text: str) -> list[str]:
    """去空白后按相邻 2 字符切。对中文是朴素 bigram，对占位足够。"""
    cleaned = "".join(ch for ch in text if not ch.isspace())
    return [cleaned[i:i+2] for i in range(0, len(cleaned)-1)]


class KeywordRetriever(Retriever):
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c
            # 按 id 入字典 = 幂等覆盖

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        q_grams = Counter(_bigrams(query))
        denom = sum(q_grams.values()) or 1
        scored: list[Retrieved] = []

        for c in self._chunks.values():
            overlap = sum((q_grams & Counter(_bigrams(c.text))).values())
            score = overlap / denom
            if score > 0:
                scored.append(Retrieved(chunk=c, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
