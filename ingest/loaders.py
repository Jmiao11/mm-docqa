"""
ingest/loaders.py —— DocumentLoader 的具体实现 + 按扩展名分派。

兑现承重墙最后一段：把"文档加载"从裸函数 parse_pdf 升格为可插拔接口。
- PdfLoader  : 委托现有 parser.parse_pdf / extract_images（不重写已在真实 PDF 上验证过的逻辑）。
- TextLoader : txt / md 纯文本加载。
- AutoLoader : 组合模式，按文件后缀分派到具体 loader。
               加新格式 = 往注册表加一项，分派逻辑零改动（开闭原则）。

routes 只持有一个 AutoLoader（经 config.build_loader 注入 app.state.loader），
对所有格式同样调用 load / load_images，无需判型 —— 上层编排格式无关。
"""
from __future__ import annotations

from pathlib import Path

from core.interfaces import Document, DocumentLoader
# 复用 parser 里通用的文本清洗与已踩平的 PDF 解析/抽图，不重复造轮子。
# 注：_clean 是 parser 的私有函数，但其逻辑（统一换行/压空行）与格式无关；
# MVP 直接复用，后续可提升为 ingest 级公共工具（见 README roadmap）。
from ingest.parser import _clean, extract_images, parse_pdf


class PdfLoader(DocumentLoader):
    """PDF 加载：复用 parser 里已在真实期刊 PDF 上踩平的解析与抽图逻辑。
    有内嵌图，故覆盖 load_images。"""

    def load(self, path: str | Path) -> Document:
        return parse_pdf(path)

    def load_images(self, path: str | Path, out_dir: str | Path) -> list[dict]:
        return extract_images(path, out_dir)


class TextLoader(DocumentLoader):
    """纯文本 / Markdown 加载：读取 → 复用 _clean 统一换行 → 单 Document。
    无分页概念：page_offsets=[0]、n_pages=1，使 page_of 一致返回第 1 页。
    Markdown 不剥语法（# / ** 等标记对 bge 语义检索影响极小，保留原文反利于引用对照）。
    无内嵌图，沿用基类 load_images 默认（返回 []）。"""

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        raw = path.read_text(encoding="utf-8", errors="replace")  # 容错解码：坏字节替换不崩
        text = _clean(raw)
        filetype = "markdown" if path.suffix.lower() == ".md" else "text"
        return Document(
            id=path.stem,
            text=text,
            source=path.name,
            metadata={"n_pages": 1, "page_offsets": [0], "filetype": filetype},
        )


class AutoLoader(DocumentLoader):
    """按扩展名分派到具体 loader。注册表是唯一真相：键=小写后缀，值=loader 实例。
    加格式只往注册表加一项，load / load_images 的分派逻辑一行不动。
    例：
    registry = {
    ".pdf": PdfLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
    }
    """

    def __init__(self, registry: dict[str, DocumentLoader]):
        self._registry = registry

    # @property的作用是让你不用加括号就能拿到这个值
    @property
    def supported(self) -> list[str]:
        """已支持的后缀（供上层做上传守卫，单一真相源）。"""
        return sorted(self._registry)

    def _pick(self, path: str | Path) -> DocumentLoader:
        suffix = Path(path).suffix.lower()
        loader = self._registry.get(suffix)
        if loader is None:
            raise ValueError(
                f"不支持的格式: {suffix or '<无后缀>'}（已支持: {self.supported}）"
            )
        return loader

    def load(self, path: str | Path) -> Document:
        return self._pick(path).load(path)

    def load_images(self, path: str | Path, out_dir: str | Path) -> list[dict]:
        return self._pick(path).load_images(path, out_dir)