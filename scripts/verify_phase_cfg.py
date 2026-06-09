"""
1.2 验收：真检索(DenseRetriever) + 真生成(LLMGenerator/DeepSeek) 端到端。
这是 Phase 1 路线图"命令行对真语料问答，出真答案、带来源"的兑现（PDF 解析在 Phase 2）。

运行前：
  1. 本地已装 openai：  pip install openai
  2. 配好环境变量 DEEPSEEK_API_KEY（key 不要写进代码）
     Windows PowerShell 临时设置：  $env:DEEPSEEK_API_KEY="你的key"
     或在系统环境变量里永久添加。
运行：  python verify_phase_cfg.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # <--- 新增这行

from core.config import RAGConfig, build_pipeline

load_dotenv()                   # <--- 新增这行：这行一执行，.env 里的 Key 就会瞬间注入到系统环境里

from chunkers.fixed import FixedSizeChunker
from core.interfaces import Document


def load_corpus(corpus_dir: str = "data/corpus") -> list[Document]:
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.txt")):
        docs.append(Document(id=p.stem, text=p.read_text(encoding="utf-8"), source=p.name))
    return docs


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("！未检测到 DEEPSEEK_API_KEY 环境变量，请先配置后再运行。")
        return

    docs = load_corpus()
    chunker = FixedSizeChunker(size=200, overlap=40)

    # 关键：把 KeywordRetriever+TemplateGenerator 换成 DenseRetriever+LLMGenerator，
    # pipeline.py 一行没改 —— 这就是接口隔离的回报。
    cfg = RAGConfig(retriever="dense", generator="llm",
                    collection_name="phase1", chunk_size=200, chunk_overlap=40)
    pipeline = build_pipeline(cfg)
    pipeline.retriever.collection.delete(where={"source": {"$ne": ""}})  # 清空旧向量 这是 ChromaDB 特有的对象，相当于关系型数据库里的一张表

    n = pipeline.index(docs)
    print(f"切块入库 {n} 块\n")

    questions = [
        "什么是检索增强生成，它解决了什么问题？",
        "为什么归一化后的向量检索要用 cosine 距离？",
        "牛顿第二定律的公式是什么？",   # 故意问语料外的，验证抗幻觉：应回答"无法回答"
    ]
    for q in questions:
        print("=" * 70)
        print(f"问：{q}\n")
        result = pipeline.run(q, k=3)
        print(result.answer)
        print(f"\n[检索命中 {len(result.retrieved)} 块，"
              f"最高分 {result.retrieved[0].score:.3f}]" if result.retrieved else "[无命中]")
        print()


if __name__ == "__main__":
    main()