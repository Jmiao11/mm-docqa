"""
DenseRetriever：语义检索。bge-small-zh-v1.5 做嵌入 + ChromaDB 做向量存储/检索。
实现 core.interfaces.Retriever，可直接替换占位的 KeywordRetriever，pipeline 不改。
"""
from __future__ import annotations

import os

from core.interfaces import Chunk, Retrieved, Retriever

# bge 的非对称检索：query 端加指令前缀，passage 端不加。
# 实验已验证：前缀通过压低不相关项得分来拉大区分度。
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

class DenseRetriever(Retriever):
    def __init__(
        self,
        embedder=None,   # 注入：外部加载好的 SentenceTransformer
        model_name: str = "BAAI/bge-small-zh-v1.5",
        persist_dir: str = "data/chroma",
        collection_name: str = "docqa"
    ):

        # 离线优先：命中本地 HF 缓存就不联网（你的环境已验证可离线加载）
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # 延迟导入：把重依赖关在类里，导入本模块不强制加载 torch/chroma
        # 把 import 锁在 __init__ 内部。这样当你在别的地方（比如 API 路由或测试脚本中）仅仅是引用了一下 dense.py 时，
        # 不会触发任何重度库的加载。只有在真正实例化 DenseRetriever() 的那一刻，才会去加载 ChromaDB。这极大地优化了系统的启动性能。
        import chromadb

        # 依赖注入：传了就共享外部实例（多检索器共用一份，省内存）；
        # 没传才自己加载（单独跑 dense.py、写小实验时方便）。
        if embedder is not None:
            self.model = embedder
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)

        # PersistentClient：向量落盘到 persist_dir，重启不丢，零运维
        self.client = chromadb.PersistentClient(path=persist_dir)
        # 关键：归一化向量必须用 cosine。Chroma 默认 L2，不改排序会错。
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- 内部：编码 ----
    # normalize_embeddings=True 的作用叫“L2 归一化”
    def _embed_passages(self, text: list[str]) -> list[list[float]]:
        vecs = self.model.encode(text, normalize_embeddings=True)
        return vecs.tolist()

    # 非对称检索
    def _embed_query(self, query: str) -> list[float]:
        vec = self.model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)
        return vec[0].tolist()

    # ---- 接口实现 ----
    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        # Chroma 的 metadata 只接受标量值；把复杂结构拍平，并补回定位信息。
        metadatas = []
        for chunk in chunks:
            md = {k: v for k, v in chunk.metadata.items()
                  if isinstance(v,(str, int, float, bool))}

            md.update({"source": chunk.source, "start": chunk.start, "end": chunk.end})
            metadatas.append(md)

        # 用 Chunk.id 作主键：相同 id 走 upsert 覆盖 = 幂等，重复入库不堆积。
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self._embed_passages([c.text for c in chunks]),
            metadatas=metadatas,
        )

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        res = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=k,
        )
        out: list[Retrieved] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for cid, text, md, dist in zip(ids, docs, metas, dists):
            md = dict(md or {})
            source = md.pop("source", "")
            start = md.pop("start", 0)
            end = md.pop("end", 0)
            chunk = Chunk(id=cid, text=text, source=source,
                          start=start, end=end, metadata=md)
            # Chroma 返回的是“距离”，cosine 距离 = 1 - 相似度。
            # 转回相似度，让 score 越大越相关，与占位检索器语义一致。

            out.append(Retrieved(chunk=chunk, score=1.0-float(dist)))

        return out


