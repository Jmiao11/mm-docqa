"""探针：看 PyMuPDF find_tables() 在课程表 PDF 上还原出的网格质量与合并单元格处理。"""
import fitz
from core.paths import UPLOAD_DIR

cands = list(UPLOAD_DIR.glob("*研究生选课*.pdf")) or list(UPLOAD_DIR.glob("*.pdf"))
print("候选 PDF:", [p.name for p in cands])
assert cands, f"{UPLOAD_DIR} 下没找到 PDF"
path = cands[0]
print("用:", path.name, "\n")

doc = fitz.open(path)
for pno, page in enumerate(doc):
    tabs = page.find_tables()
    print(f"==== 第 {pno} 页：检测到 {len(tabs.tables)} 个表 ====")
    for ti, t in enumerate(tabs.tables):
        grid = t.extract()
        ncol = max((len(r) for r in grid), default=0)
        print(f"\n-- 表 {ti}: {len(grid)} 行 x {ncol} 列 --")
        for ri, row in enumerate(grid):
            cells = [(c or "∅").replace("\n", "/") for c in row]
            print(f"  行{ri}: {cells}")
doc.close()