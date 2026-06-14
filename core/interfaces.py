"""
项目契约层：所有数据结构和组件接口。
其余代码（pipeline、各实现、API、评估）只依赖这里，不互相依赖。
改实现不改这里；这里一旦稳定，后续 Phase 全是"挂"上来，不是"改"出来。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# 数据类（5 个）：在组件之间流动的"货物"
# ============================================================
@dataclass
class Document:
    """一篇原始文档：解析之后、分块之前的状态。"""
    id: str
    text: str
    source: str  # 来源标识（文件名/路径），引用链的起点
    metadata: dict[str, Any] = field(default_factory=dict)
    # default_factory=dict：意思是“默认值工厂”。它接收一个函数（这里传的是内置函数 dict）。
    # 每当你创建一个新的 Document 实例时，Python 就会在幕后自动调用一次这个函数（相当于执行 dict()），
    # 从而为当前实例生成一个全新的、干净的空字典。


@dataclass
class Chunk:
    """文档切出的一块。id 确定性，携带原文定位 (start,end) 与 metadata。"""
    id: str
    text: str
    source: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, start: int, end: int) -> str:
        """确定性 id：同一 (source,start,end) 永远得到同一 id。
        —— 重复入库可覆盖去重；引用来源可追溯；评估可对齐。"""
        raw = f"{source},{start},{end}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Retrieved:
    """一次检索命中：块本身 + 相关性分数。检索器返回的是它，不是纯文本。"""
    chunk: Chunk
    score: float

@dataclass
class DeleteResult:
    """按 source 删除的产物：删了几块向量 + 这些块附带的图片文件路径。
    image_paths 让上层（routes 编排）能连带删掉 data/images 下的物理文件，
    避免孤儿文件。由 retriever 在删除时一次性带出（查与删原子，无竞态）。"""
    n_chunks: int = 0
    image_paths: list[str] = field(default_factory=list)


@dataclass
class RAGResult:
    """一次完整问答的产物：同时保留检索中间结果和最终答案。
    —— 让上层能分层归因：答得差，是检索没召回到，还是生成没用好上下文。"""
    query: str
    retrieved: list[Retrieved]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSample:
    """评估样本（背书层用）：问题 + 期望答案 / 期望命中的块 id。"""
    query: str
    expected_answer: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)


# ============================================================
# 接口（5 个 ABC）：组件的"插座"，定义能力不定义实现
# ============================================================
# path -> Loader -> Document -> Chunker -> Retriever -> Generator
class DocumentLoader(ABC):
    """加载器：原始文件 path -> Document（解析 + 清洗 + 元数据）。
    处于整条链最上游：path -> Loader -> Document -> Chunker -> Retriever -> Generator。
    是承重墙最后一段——把"文档加载"从裸函数升格为可插拔接口，与下游三层对称。

    load_images 是"可选能力"而非抽象方法：多数格式（txt/md）无内嵌图，基类默认
    返回空列表；有图格式（PDF）覆盖它。这样上层编排对所有格式同样调用、无需判型。"""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """把单个文件解析成一个 Document。"""
        ...

    # 软契约
    def load_images(self, path: str | Path, out_dir: str | Path) -> list[dict]:
        """抽取文档内嵌图，返回 [{page,xref,path,width,height}, ...]。默认无图。"""
        return []


class Chunker(ABC):
    """分块器：Document -> list[Chunk]"""

    @abstractmethod
    def split(self, docs: Document) -> list[Chunk]:
        ...


class Retriever(ABC):
    """检索器：建索引 + 按 query 取回 top-k。"""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """把块建入索引。约定幂等：相同 Chunk.id 覆盖而非重复堆积。"""
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        """返回相关性降序的前 k 个 Retrieved。"""
        ...


class Generator(ABC):
    """生成器：query + 检索上下文 -> 答案文本。
    注意 contexts 是 list[Retrieved] 而非 list[str]——这样未来 vlm
    才能从 chunk.metadata 里拿到图片路径，签名不用改。"""

    @abstractmethod
    def generate(self, query: str, context: list[Retrieved]) -> str:
        ...


class Evaluator(ABC):
    """评估器（背书层）：用 RAGResult + EvalSample 算指标。"""

    @abstractmethod
    def evaluate(self, results: list[RAGResult], samples: list[EvalSample]) -> dict[str, float]:
        ...