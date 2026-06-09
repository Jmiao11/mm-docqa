"""
chunkers/semantic.py —— 句界 + 段落硬边界的分块器。

相比 fixed.py 的纯字符滑窗，两条改进：
  1) 以句子为最小不可分单位累积成块 —— 句子不再被从中间切碎。
  2) 段落空行(\n\n)是硬边界，贪心打包绝不跨段 —— 正文段与"参考文献"段
     被强制分到不同块，缓解"正文/引用混进同一块"的污染。

定位：只缓解"混"，不删除参考文献块（那是 Phase 5）。彻底治污染靠
semantic(本块) + rerank(把正文顶到引用之上) + Phase5(剥离参考文献) 三招叠加。

契约不变：仍产出带准确 (start,end) 原文字符跨度的 Chunk，make_id 确定性、
引用可定位。实现上用 finditer 保留绝对位置，绝不用 re.split（会丢偏移量）。
"""
from __future__ import annotations

import re

from core.interfaces import Chunk, Chunker, Document

_SENT_END = re.compile(r"[。！？；\.\?!\n]+")   # 句子终止符
_PARA_BREAK = re.compile(r"\n\s*\n")            # 段落硬边界：空行


class SemanticChunker(Chunker):
    def __init__(self, max_size: int = 300, overlap_sentences: int = 1):
        assert max_size > 0
        assert overlap_sentences >= 0
        self.max_size = max_size
        self.overlap_sentences = overlap_sentences

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """全文 → 句子的 (start,end) 绝对位置；空白句跳过，位置照常推进。"""
        spans: list[tuple[int, int]] = []
        start = 0
        for m in _SENT_END.finditer(text):
            end = m.end()
            if text[start:end].strip():
                spans.append((start, end))
            start = end
        if start < len(text) and text[start:].strip():
            spans.append((start, len(text)))
        return spans

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        spans = self._sentence_spans(text)
        if not spans:
            return []

        # 每个句子归属哪个段落：落在它前面的空行越多，段号越大
        # breaks（空行）：就是公路上的收费站。
        # spans（句子）：就是行驶在公路上的汽车。
        breaks = [m.start() for m in _PARA_BREAK.finditer(text)]
        para_id = [sum(1 for b in breaks if b < s) for s, _ in spans]
        # para_id 就会瞬间生成一个与句子一一对应的数组（比如 [0, 0, 1, 1, 1, 2...]）

        chunks: list[Chunk] = []
        n = len(spans)
        i = 0

        while i < n:
            cur_start = spans[i][0]
            cur_para = para_id[i]
            j = i
            while j < n and para_id[j] == cur_para and (spans[j][1] - cur_start) <= self.max_size:
                j += 1
            if j == i:                       # 单句超长：至少收一句，防死循环
                j = i + 1
            end = spans[j - 1][1]
            piece = text[cur_start:end]

            if piece.strip():
                chunks.append(Chunk(
                    id=Chunk.make_id(doc.source, cur_start, end),
                    text=piece, source=doc.source,
                    start=cur_start, end=end,
                    metadata=dict(doc.metadata),
                ))

            stopped_by_para = j < n and para_id[j] != cur_para
            if stopped_by_para or self.overlap_sentences == 0:
                i = j                        # 跨段不重叠，下一块干净开始
            else:
                next_i = j - self.overlap_sentences
                i = next_i if next_i > i else j
        return chunks