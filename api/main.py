"""
api/main.py —— FastAPI 入口。
lifespan 里把 pipeline 和 db 初始化一次，存进 app.state 全局共享。
昂贵资源（bge 模型）整个服务生命周期只加载一次。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import router
from core.config import RAGConfig, build_pipeline
from store.metadata_db import MetadataDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 启动时：初始化共享资源（只跑一次）===
    cfg = RAGConfig(
        retriever="rerank",  # Phase 3：dense+BM25(RRF) 召回 → cross-encoder 重排
        chunker="semantic",  # Phase 3：句界+段落切分，不切碎句子
        generator="llm",
        collection_name="docqa_v2",  # 新集合：旧 docqa 是 fixed 切的向量，不混
    )
    app.state.pipeline = build_pipeline(cfg)     # 这里加载 bge，仅此一次
    from core.paths import DB_PATH
    app.state.db = MetadataDB(str(DB_PATH))

    app.state.errors = {}                        # doc_id -> 错误信息

    # 多模态：有 Moonshot key 才启用图 caption；没有则降级为纯文本入库
    import os
    from ingest.captioner import Captioner
    app.state.captioner = Captioner() if os.environ.get("MOONSHOT_API_KEY") else None
    print(f"[main] 共享资源已初始化：pipeline + db + captioner="
          f"{'on' if app.state.captioner else 'off'}")

    print("[main] 共享资源已初始化：pipeline + db")
    yield
    # === 关闭时：清理 ===
    app.state.db.close()
    print("[main] 已清理资源")


app = FastAPI(title="mm-docqa", description="多模态文档问答助手", lifespan=lifespan)
app.include_router(router)


@app.get("/")
async def root():
    return {"service": "mm-docqa", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)