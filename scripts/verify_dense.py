"""
1.1 验收：在本地用真 bge 模型跑 DenseRetriever，并和 Phase 0 的 KeywordRetriever 对比。
从项目根运行：  python verify_dense.py

看点：找一个"语义相关但字面不重叠"的问题，dense 能召回、keyword 召回差，
即可证明语义检索的价值。
"""
from __future__ import annotations

from pathlib import Path

from chunkers.fixed import FixedSizeChunker
from core.interfaces import Document
from retrievers.dense import DenseRetriever
from retrievers.keyword import KeywordRetriever


def load_corpus(corpus_dir: str = "data/corpus") -> list[Document]:
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.txt")):
        docs.append(Document(id=p.stem, text=p.read_text(encoding="utf-8"), source=p.name))
    return docs


def show(title, retriever, query, k=3):
    print(f"\n--- {title} | 问题：{query} ---")
    hits = retriever.retrieve(query, k)
    if not hits:
        print("  （无命中）")
    for i, r in enumerate(hits, 1):
        snip = r.chunk.text.strip().replace("\n", " ")[:50]
        print(f"  [{i}] score={r.score:.3f} {r.chunk.source} | {snip}…")


def main():
    docs = load_corpus()
    chunks = []
    chunker = FixedSizeChunker(size=200, overlap=40)
    for d in docs:
        chunks.extend(chunker.split(d))
    print(f"语料 {len(docs)} 篇，切块 {len(chunks)} 块")

    # 每次重跑用全新集合，避免旧向量干扰对比
    dense = DenseRetriever(persist_dir="../data/chroma", collection_name="verify_dense")
    dense.collection.delete(where={"source": {"$ne": ""}})
    # 这是因为 ChromaDB（以及很多现代数据库）非常死板，它的 delete 操作强制要求你必须给出一个“条件（where）”
    # 它拒绝执行没有任何条件的“无差别屠杀”，以防程序员手滑把整个数据库删没了。
    # 于是，机智（且狡猾）的程序员就想出了一个绕过这个死板规定的办法。

    dense.index(chunks)

    keyword = KeywordRetriever()
    keyword.index(chunks)

    # 故意挑字面不重叠、但语义相关的问法
    for q in [
        "怎么让大模型回答时引用资料出处",   # 字面没"RAG/引用来源"，但语义指向 RAG
        "存向量并快速找最像的那条用什么",   # 字面没"向量数据库/最近邻"
    ]:
        print("=" * 70)
        show("Dense(语义)", dense, q)
        show("Keyword(字面)", keyword, q)


if __name__ == "__main__":
    main()
