"""固定字符窗口分块，带重叠。占位实现，但产出的 Chunk 是合规的。"""
from __future__ import annotations

from core.interfaces import Chunk, Chunker, Document


class FixedSizeChunker(Chunker):
    def __init__(self, size: int = 300, overlap: int = 50):
        assert overlap < size,  "overlap 必须小于 size，否则窗口不前进会死循环"
        self.size = size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for start in range(0, len(text), step):
            end = min(start + self.size, len(text))
            piece = text[start:end]
            if piece.strip():
                chunks.append(Chunk(
                    id=Chunk.make_id(doc.source, start, end),
                    text=piece,
                    source=doc.source,
                    start=start,
                    end=end,
                    metadata=dict(doc.metadata),
                    # 拷贝一份，避免多块共享同一 dict
                ))

            if end == len(text):
                break
        return chunks
