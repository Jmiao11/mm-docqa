"""
core/config.py —— 配置驱动的 pipeline 装配工厂。

职责：把"用哪些组件、什么参数"从业务代码里抽出来，集中成一个配置对象  一个工厂函数。
业务代码永远是两行：
    cfg = RAGConfig(retriever="dense")
    pipeline = build_pipeline(cfg)
想换检索策略，只改 cfg.retriever 这个字符串，build_pipeline 的调用方一行不动。

这是本项目"系统设计"训练的核心落点：关注点分离  依赖注入归位  开闭原则。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.pipeline import RAGPipeline


@dataclass
class RAGConfig:
    """一次运行的全部可调参数。新增策略时，往这里加字段即可。"""
    # —— 分块 ——
    chunker: str = "fixed"  # "fixed" | "semantic"
    chunk_size: int = 200  # fixed: 窗口大小 / semantic: 块字符上限
    chunk_overlap: int = 40  # fixed 用：字符级重叠
    chunk_overlap_sentences: int = 1  # semantic 用：块间重叠句数

    # —— 检索 ——
    retriever: str = "dense"          # "keyword" | "dense"（Phase 3 再加 hybrid/rerank）
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    persist_dir: str = "data/chroma"
    collection_name: str = "docqa"
    top_k: int = 4

    rrf_k: int = 60  # hybrid: RRF 平滑常数(原论文经验值)
    candidate_k: int = 20  # hybrid: 每路召回的候选数(>最终 k)
    rerank_model: str = "BAAI/bge-reranker-base"

    # —— 生成 ——
    generator: str = "llm"            # "template"（占位） | "llm"（DeepSeek）
    llm_model: str = "deepseek-v4-flash"

    # 杂项扩展位：不想为每个小参数都加字段时，丢这里
    extra: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 昂贵资源的集中加载：bge 模型只在这里 new 一次，注入给需要它的检索器。
# 用一个简单缓存避免同一进程内重复加载（dense 和将来的 hybrid/rerank 共享同一份）。
# ============================================================
_embedder_cache: dict[str, Any] = {}

def get_embedder(model_name: str):
    """按模型名缓存：第一次加载，之后命中缓存直接返回同一实例。"""
    if model_name not in _embedder_cache:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        print(f"[config] 加载嵌入模型 {model_name} …（仅首次）")
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]

# (b) get_embedder 之后，加 reranker 缓存(和 bge 同样的"加载一次、共享"思路)
_reranker_cache: dict[str, Any] = {}

def get_reranker(model_name: str):
    """重排模型(CrossEncoder)同样缓存复用，全进程加载一次。"""
    if model_name not in _reranker_cache:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        # 新版 transformers 在 offline 下仍会对仓库名发起 model_info 网络探测并崩溃；
        # 先解析成本地缓存目录，CrossEncoder 见到本地路径即跳过该探测。
        model_path = model_name
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_name, local_files_only=True)
        except Exception:
            pass
        print(f"[config] 加载重排模型 {model_name} …（仅首次）")
        _reranker_cache[model_name] = CrossEncoder(model_path)
    return _reranker_cache[model_name]


# ============================================================
# 三个装配函数：各管一层，按 cfg 选择实现。
# 加新策略 = 在对应函数里加一个分支，不动别处。这就是"开闭原则"。
# ============================================================
def _build_chunker(cfg: RAGConfig):
    if cfg.chunker == "fixed":
        from chunkers.fixed import FixedSizeChunker
        return FixedSizeChunker(size=cfg.chunk_size, overlap=cfg.chunk_overlap)

    if cfg.chunker == "semantic":
        from chunkers.semantic import SemanticChunker
        return SemanticChunker(max_size=cfg.chunk_size, overlap_sentences=cfg.chunk_overlap_sentences)

    raise ValueError(f"未知 chunker: {cfg.chunker}")


def _build_retriever(cfg: RAGConfig):
    if cfg.retriever == "keyword":
        from retrievers.keyword import KeywordRetriever
        return KeywordRetriever()

    if cfg.retriever == "dense":
        from retrievers.dense import DenseRetriever
        from core.paths import CHROMA_DIR
        embedder = get_embedder(cfg.embed_model)
        return DenseRetriever(
            embedder=embedder,
            persist_dir=str(CHROMA_DIR),          # 绝对路径，锚定项目根，不随 cwd 变
            collection_name=cfg.collection_name,
        )

    if cfg.retriever == "hybrid":
        from retrievers.hybrid import HybridRetriever
        from retrievers.dense import DenseRetriever
        from core.paths import CHROMA_DIR
        embedder = get_embedder(cfg.embed_model)
        dense = DenseRetriever(embedder=embedder, persist_dir=str(CHROMA_DIR), collection_name = cfg.collection_name)
        return HybridRetriever(dense, k_rrf=cfg.rrf_k, candidate_k=cfg.candidate_k)

    # (c) _build_retriever 里，hybrid 分支后面加 rerank 分支
    if cfg.retriever == "rerank":
        from retrievers.rerank import RerankRetriever
        from retrievers.hybrid import HybridRetriever
        from retrievers.dense import DenseRetriever
        from core.paths import CHROMA_DIR
        embedder = get_embedder(cfg.embed_model)
        dense = DenseRetriever(embedder=embedder, persist_dir=str(CHROMA_DIR), collection_name=cfg.collection_name)
        hybrid = HybridRetriever(dense, k_rrf=cfg.rrf_k, candidate_k=cfg.candidate_k)
        reranker = get_reranker(cfg.rerank_model)
        return RerankRetriever(base=hybrid, reranker=reranker, candidate_k=cfg.candidate_k)

    raise ValueError(f"未知 retriever: {cfg.retriever}")



def _build_generator(cfg: RAGConfig):
    if cfg.generator == "template":
        from generators.template import TemplateGenerator
        return TemplateGenerator()

    if cfg.generator == "llm":
        from generators.llm import LLMGenerator
        return LLMGenerator(model=cfg.llm_model)

    if cfg.generator == "vlm":                          # ← 新增这 4 行
        from generators.vlm import VLMGenerator
        # 文本路径仍用 DeepSeek(cfg.llm_model)；命中图块时内部切 kimi-k2.6 看图作答
        return VLMGenerator(model=cfg.llm_model)

    raise ValueError(f"未知 generator: {cfg.generator}")





def build_pipeline(cfg: RAGConfig) -> RAGPipeline:
    """工厂主入口：读配置 → 组装三层 → 返回 pipeline。业务代码只调它。"""
    return RAGPipeline(
        chunker=_build_chunker(cfg),
        retriever=_build_retriever(cfg),
        generator=_build_generator(cfg),
    )
