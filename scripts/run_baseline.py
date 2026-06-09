"""
Phase 0 验收脚本：用占位实现端到端跑通 分块->检索->生成。
从项目根目录运行：  python run_baseline.py
"""
from __future__ import annotations

from pathlib import Path

from chunkers.fixed import FixedSizeChunker
from core.interfaces import Document
from core.pipeline import RAGPipeline
from generators.template import TemplateGenerator
from retrievers.keyword import KeywordRetriever


def load_corpus(corpus_dir: str = "data/corpus") -> list[Document]:
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.txt")):
        docs.append(Document(id=p.stem, text=p.read_text(encoding="utf-8"), source=p.name))
        # p.stem 自动获取不带后缀的文件名（如 "rag"）
        # p.name 自动获取带后缀的完整文件名（如 "rag.txt"）
    return docs


def main():
    docs = load_corpus()
    print(f"加载 {len(docs)} 篇文档")

    # 组装 pipeline：三层全是占位实现，但接口契约和真实现完全一致
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(size=200, overlap=40),
        retriever=KeywordRetriever(),
        generator=TemplateGenerator(),
    )

    n = pipeline.index(docs)
    print(f"切块入库 {n} 块\n")

    for q in ["什么是检索增强生成", "向量数据库有什么作用"]:
        print("=" * 64)
        result = pipeline.run(q, k=3)
        print(result.answer)
        # RAGResult 同时握有 retrieved 和 answer —— 这就是分层归因的本钱
        print(f"\n[命中 {len(result.retrieved)} 块; 最高分 "
              f"{result.retrieved[0].score:.3f}]" if result.retrieved else "[无命中]")
        print()


if __name__ == "__main__":
    main()
