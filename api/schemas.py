"""
api/schemas.py —— 请求/响应的数据形状（Pydantic 模型），即前后端的合同。
FastAPI 会用它们自动校验入参、自动生成 /docs 文档。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- 上传相关 ----------
class UploadResponse(BaseModel):
    """上传后立刻返回（异步：不等入库完成）。"""
    doc_id: str
    source: str
    status: str = "pending"            # pending → processing → indexed / failed


class DocStatus(BaseModel):
    """单个文档的状态（轮询 GET /documents/{id}/status 用）。"""
    doc_id: str
    source: str
    status: str
    n_pages: int = 0
    n_chunks: int = 0
    error: str = ""                    # 失败时填原因，给前端反馈


class DocInfo(BaseModel):
    """文档列表项（GET /documents 用）。"""
    doc_id: str
    source: str
    status: str
    n_pages: int = 0
    n_chunks: int = 0


# ---------- 提问相关 ----------
class QueryRequest(BaseModel):
    """提问入参。"""
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(4, ge=1, le=20, description="检索取回的块数")
    session_id: str = Field("default", description="会话标识，用于历史记录")


class Citation(BaseModel):
    """单条引用来源。结构和 llm.py 里 collect_citations 产出的一致。"""
    n: int                             # 答案里的编号 [n]
    id: str                            # chunk.id
    source: str                        # 来源文件名
    score: float
    text: str = ""                     # 检索到的原文片段（可溯源：点击展开看依据）
    kind: str = "text"                 # text | image（图块的 text 是 VLM caption）


class ImageHit(BaseModel):
    """命中且被引用的图块，供前端 Gallery 展示。path 是后端本机绝对路径。"""
    n: int
    id: str
    source: str
    path: str


class QueryResponse(BaseModel):
    """提问返回：答案 + 结构化引用 + 命中块数。"""
    answer: str
    citations: list[Citation] = []
    images: list[ImageHit] = []  # ← 新增
    n_retrieved: int = 0
    standalone_query: str = ""   # 多轮：历史感知改写后的独立检索 query（首轮=原问题）


# ---------- 会话 ----------
class SessionInfo(BaseModel):
    """会话列表项（从 messages 聚合派生）。"""
    session_id: str
    title: str = ""
    n: int = 0
    last_at: float = 0.0


class MessageItem(BaseModel):
    """单条历史消息（切换会话时还原显示用）。"""
    role: str
    content: str
    sources: list[Citation] = []


class SessionDeleteResponse(BaseModel):
    """删除会话返回。"""
    session_id: str
    deleted_messages: int = 0


# ---------- 通用 ----------
class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    n_chunks: int = 0        # 删除的向量块数（含文本块+图块）
    n_images: int = 0        # 删除的物理图片文件数