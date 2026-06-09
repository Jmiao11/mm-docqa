"""
RAGPipeline：串联 分块 -> 检索 -> 生成。
只依赖 core.interfaces 的三个 ABC，不 import 任何具体实现。
这是"换任一层都不动主流程"承诺的兑现处。
在 pipeline.py 的世界里，它根本不知道、也不想知道你底层用的是 ChromaDB 还是 DeepSeek。
它手里只有一张从 interfaces.py 批发来的“合格供应商清单”（抽象基类）。只要你送进来的组件符合清单要求，它就闭着眼睛按部就班地拉动流水线。
"""

from __future__ import annotations

from core.interfaces import (
    Chunker,
    Document,
    Generator,
    RAGResult,
    Retriever,
)
class RAGPipeline:
    def __init__(self, chunker: Chunker, retriever: Retriever, generator: Generator):
        # 类型标注写的是接口，不是某个实现 —— 依赖倒置
        # 我不管你传进来的具体实例是什么，只要它是 Chunker/Retriever/Generator 的子类就行
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator

    def index(self, docs: list[Document]) -> int:
        """建索引：文档 -> 块 -> 入库。返回入库块数。"""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunker.split(doc))
            # 在 Python 中，extend() 是列表（list）对象的一个内置方法，用来把另一个可迭代对象（通常是另一个列表）中的所有元素，逐个添加进当前列表的末尾。

        self.retriever.index(all_chunks)
        return len(all_chunks)

    def run(self, query: str, k: int = 4) -> RAGResult:
        """一次问答：检索 -> 生成 -> 打包成 RAGResult。"""
        retrieved = self.retriever.retrieve(query, k)
        answer = self.generator.generate(query, retrieved)
        return RAGResult(query, retrieved, answer)
