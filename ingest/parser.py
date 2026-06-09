"""
ingest/parser.py —— 用 PyMuPDF 把 PDF 解析成 Document。
处于整条链最上游：PDF → parser → Document → Chunker → ...
Phase 2.1 只做文本（图在 Phase 4）。清洗做最低限度，不过度工程。
"""
from __future__ import annotations

import re
from pathlib import Path

from core.interfaces import Document


def _clean(text: str) -> str:
    """最低限度清洗：统一换行、合并 3+ 连续空行为 1 个、去首尾空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n") # 这行代码暴力且有效地把所有奇奇怪怪的换行符，全部统一转换成了最标准的 \n。这为后续的正则匹配扫清了障碍。
    text = re.sub(r"\n{3,}", "\n\n", text)        # 多个空行压成一个
    text = re.sub(r"[ \t]+\n", "\n", text)        # 行尾空格去掉：很多人在写文档时，习惯在一段话敲完后，多打几个空格再按回车。
    return text.strip()


def parse_pdf(path: str | Path) -> Document:
    """把一个 PDF 解析成单个 Document。
    metadata 里记录：总页数 + 每页在拼接全文中的起始字符位置（page_offsets）。
    page_offsets 让我们之后能由"字符位置"反查"第几页"，用于引用展示。"""
    import fitz  # PyMuPDF；延迟导入，避免没装时整个模块报错

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在: {path}")

    doc = fitz.open(path)
    parts: list[str] = []
    page_offsets: list[int] = []      # page_offsets[i] = 第 i 页正文在全文中的起始位置
    cursor = 0
    for page in doc:
        page_text = _clean(page.get_text())
        page_offsets.append(cursor)
        parts.append(page_text)
        cursor += len(page_text) + 2  # +2 是下面用 "\n\n" 连接的两个字符
    doc.close()

    full_text = "\n\n".join(parts)
    return Document(
        id=path.stem,
        text=full_text,
        source=path.name,
        metadata={
            "n_pages": len(page_offsets),
            "page_offsets": page_offsets,
            "filetype": "pdf",
        },
    )


def extract_images(path: str | Path, out_dir: str | Path,
                   min_w: int = 150, min_h: int = 100, zoom: float = 3.0) -> list[dict]:
    """抽取 PDF 里的图表，存为 PNG，返回每张图的元信息。

    策略要点（都是真 PDF 上踩出来的）：
    - 图表是嵌入位图，用 get_images 定位；get_drawings 在本类期刊里全是表格线/版式
      线（每页近百条），是噪声，不用。
    - 按【原生尺寸】过滤掉 logo/公式小图（默认 min 150x100）。
    - 不直接抽 xref 原始位图：嵌入图常带翻转/旋转变换矩阵，原始位图是镜像的，会让
      VLM 读错图中文字。改为【按图在页面上的位置 bbox 渲染】，拿到读者看到的正确朝向。
    - 文件名含 xref，确定性命名 → 重复抽取幂等。
    返回 [{page, xref, path, width, height}, ...]；路径供 captioner 生成 caption，
    写进 Chunk.metadata（契约早预留），实现 caption-then-embed。
    """
    import fitz

    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(path)
    figures: list[dict] = []
    seen: set[int] = set()
    for pno, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref, w, h = img[0], img[2], img[3]   # 原生宽高在元组里，无需建 Pixmap
            if xref in seen:
                continue
            seen.add(xref)
            if w < min_w or h < min_h:            # 原生尺寸过滤装饰小图
                continue
            rects = page.get_image_rects(xref)    # 图在页面上的放置矩形
            if not rects:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rects[0])

            fpath = out_dir / f"{path.stem}_p{pno}_x{xref}.png"
            pix.save(str(fpath))
            figures.append({"page": pno, "xref": xref, "path": str(fpath),
                            "width": pix.width, "height": pix.height})
            pix = None

    doc.close()
    return figures

def page_of(offset: int, page_offsets: list[int]) -> int:
    """给定字符位置，反查它在第几页（从 1 开始）。供引用展示用。"""
    page = 1
    for i, start in enumerate(page_offsets):
        if offset >= start:
            page = i + 1
        else:
            break
    return page
