"""3.1 验收：在真实 PDF 上对比 fixed vs semantic 分块。无网络。
运行：python verify_semantic.py"""
from pathlib import Path
from ingest.parser import parse_pdf
from chunkers.fixed import FixedSizeChunker
from chunkers.semantic import SemanticChunker

# 自动找项目里的 PDF（根目录 / data / data/uploads 都扫）
cands = list(Path("..").glob("*.pdf")) + list(Path("../data").rglob("*.pdf"))
assert cands, "没找到 PDF，请确认测试 PDF 在项目里"
doc = parse_pdf(str(cands[0]))
print(f"PDF: {doc.source} | 全文 {len(doc.text)} 字 | {doc.metadata['n_pages']} 页\n")

SENT_END = set("。！？；.?!\n")
bound = doc.text.find("参考文献")   # 没有就是 -1，straddle 判据自动失效

def analyze(name, chunks):
    offset_ok = all(doc.text[c.start:c.end] == c.text for c in chunks)
    mid_cut = sum(1 for c in chunks if c.end < len(doc.text) and doc.text[c.end-1] not in SENT_END)
    straddle = sum(1 for c in chunks if bound > 0 and c.start < bound < c.end)
    print(f"[{name:16}] 块数={len(chunks):>3} | 偏移全对={offset_ok} | 句子被切碎={mid_cut:>3} | 跨参考文献边界={straddle}")

analyze("fixed",    FixedSizeChunker(size=200, overlap=40).split(doc))
analyze("semantic", SemanticChunker(max_size=200, overlap_sentences=1).split(doc))
print("\n预期：semantic 的'句子被切碎'和'跨参考文献边界'都应明显低于 fixed。")