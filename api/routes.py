"""
api/routes.py —— 五个 HTTP 接口，把 parser/db/pipeline 串成后端服务。
异步入库：上传只记 pending + 挂后台任务立刻返回；后台跑 parse→index→回填。
路由不自己造资源，从 request.app.state 取共享的 pipeline / db。
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile

# ① 顶部 import
from api.schemas import (
    Citation, DeleteResponse, DocInfo, DocStatus, ImageHit,   # ← 加 ImageHit
    QueryRequest, QueryResponse, UploadResponse,
)
from ingest.parser import parse_pdf

router = APIRouter()

from core.paths import UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 后台任务：真正的入库工作在这里，脱离请求线程 ----------
def _index_document(app, doc_id: str, source: str, file_path: str) -> None:
    """解析 → 切块入库 → 回填状态。出错必须把 failed 写回 db（异步必修课）。"""
    db = app.state.db
    pipeline = app.state.pipeline
    loader = app.state.loader
    try:
        db.update_document(doc_id, status="processing")

        doc = loader.load(file_path)  # 按扩展名分派：pdf/txt/md，routes 不判型
        doc.id = doc_id  # 用统一的 doc_id，便于关联
        n_text = pipeline.index([doc])  # 文本：semantic 切块入库

        # 多模态：抽图 → VLM caption → 图块入同一检索器（caption-then-embed）
        n_img = 0
        captioner = getattr(app.state, "captioner", None)
        if captioner is not None:
            from ingest.captioner import build_image_chunks
            from core.paths import DATA_DIR
            figures = loader.load_images(file_path, DATA_DIR / "images")
            img_chunks = build_image_chunks(figures, captioner, doc.source)
            if img_chunks:
                pipeline.retriever.index(img_chunks)
                n_img = len(img_chunks)

        db.update_document(doc_id, status="indexed",
                           n_pages=doc.metadata.get("n_pages", 0),
                           n_chunks=n_text + n_img)
    except Exception as e:
        # 后台没人看着，错误必须落库，否则前端永远 pending
        traceback.print_exc()
        db.update_document(doc_id, status="failed")
        app.state.errors[doc_id] = str(e)      # 错误详情存内存，供 status 接口读


# ---------- 1) 上传：存文件 + 记 pending + 挂后台任务 + 立刻返回 ----------
@router.post("/documents", response_model=UploadResponse)
async def upload_document(request: Request, background: BackgroundTasks,
                          file: UploadFile):
    loader = request.app.state.loader
    suffix = Path(file.filename).suffix.lower()
    if suffix not in loader.supported:  # 已支持后缀以 loader 注册表为唯一真相
        raise HTTPException(400, f"暂不支持 {suffix or '该'} 格式；已支持：{loader.supported}")

    doc_id = Path(file.filename).stem
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:                 # 把上传的文件存到磁盘
        shutil.copyfileobj(file.file, f)

    db = request.app.state.db
    db.add_document(doc_id, file.filename, status="pending")

    # 关键：挂后台任务，函数立刻返回，不等入库
    background.add_task(_index_document, request.app, doc_id, file.filename, str(dest))
    return UploadResponse(doc_id=doc_id, source=file.filename, status="pending")


# ---------- 2) 查状态：前端轮询直到 indexed ----------
@router.get("/documents/{doc_id}/status", response_model=DocStatus)
async def document_status(request: Request, doc_id: str):
    db = request.app.state.db
    d = db.get_document(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return DocStatus(
        doc_id=d["id"], source=d["source"], status=d["status"],
        n_pages=d["n_pages"], n_chunks=d["n_chunks"],
        error=request.app.state.errors.get(doc_id, ""),
    )


# ---------- 3) 列表 ----------
@router.get("/documents", response_model=list[DocInfo])
async def list_documents(request: Request):
    db = request.app.state.db
    return [
        DocInfo(doc_id=d["id"], source=d["source"], status=d["status"],
                n_pages=d["n_pages"], n_chunks=d["n_chunks"])
        for d in db.list_documents()
    ]


# ---------- 4) 提问：调 pipeline，返回答案 + 引用 ----------
@router.post("/query", response_model=QueryResponse)
async def query(request: Request, req: QueryRequest):
    pipeline = request.app.state.pipeline
    db = request.app.state.db

    result = pipeline.run(req.question, k=req.k)

    # 从 llm 生成的答案里解析引用（复用 generator 的逻辑）
    citations: list[Citation] = []
    images: list[ImageHit] = []  # ← 新增
    gen = pipeline.generator

    if hasattr(gen, "build_context") and hasattr(gen, "collect_citations"):
        _, mapping = gen.build_context(result.retrieved)
        for c in gen.collect_citations(result.answer, mapping):
            citations.append(Citation(**c))

        # 命中且被引用的图块 → 前端 Gallery（与引用编号严格对齐）
        # 校验文件存在，与 VLMGenerator._image_contexts 一致，避免展示坏图
        from pathlib import Path as _Path
        for c in citations:
            r = mapping.get(c.n)
            if r is None:
                continue
            md = r.chunk.metadata or {}
            p = md.get("image_path")
            if md.get("kind") == "image" and p and _Path(p).exists():
                images.append(ImageHit(n=c.n, id=r.chunk.id,
                                       source=r.chunk.source, path=p))
    # 记会话历史
    db.add_message(req.session_id, "user", req.question)
    db.add_message(req.session_id, "assistant", result.answer,
                   sources=[c.model_dump() for c in citations])


    return QueryResponse(answer=result.answer, citations=citations,
                         images=images, n_retrieved=len(result.retrieved))   # ← 加 images=images

# ---------- 5) 删除 ----------
def _safe_unlink(path) -> int:
    """删一个物理文件，尽力而为：存在则删、返回 1；不存在或删失败 → 记日志返回 0，
    绝不抛出。删图片和删原始 PDF 共用——都属于"副作用文件，失败不影响检索正确性"。"""
    try:
        fp = Path(path)
        if fp.exists():
            fp.unlink()
            return 1
    except Exception as e:
        traceback.print_exc()
        print(f"[delete] 物理文件删除失败（已跳过，不影响检索）: {path} — {e}")
    return 0


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(request: Request, doc_id: str):
    """删除文档：入库的逆操作，对称拆掉入库写下的三处（向量 / 图片文件 / 元数据）。

    顺序锁死：先删向量（顺手带出图片路径，趁向量还在）→ 删图片文件 → 删元数据。
    失败分级（无跨存储事务）：删向量是主操作，影响检索正确性，失败则整体报错；
    删图片文件是副作用，孤儿文件只是磁盘垃圾、不影响正确性，失败只记日志不中断。
    """
    db = request.app.state.db
    pipeline = request.app.state.pipeline

    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    source = doc["source"]

    # ① 主操作：删向量（retriever 按 source 真删 + 重建 BM25 + 带出图片路径）
    result = pipeline.retriever.delete_by_source(source)

    # ② 副作用：删物理文件（抽出的图 + 原始 PDF），尽力而为，失败不中断
    n_images = sum(_safe_unlink(p) for p in result.image_paths)
    _safe_unlink(UPLOAD_DIR / source)  # 原始 PDF：入库时存在 UPLOAD_DIR/{source}

    # ③ 删元数据（本就支持）
    deleted = db.delete_document(doc_id)

    return DeleteResponse(doc_id=doc_id, deleted=deleted,
                          n_chunks=result.n_chunks, n_images=n_images)