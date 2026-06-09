"""验收抽图：python scripts/verify_extract.py，然后去 data/images 打开图肉眼确认。"""
import glob
from core.paths import DATA_DIR
from ingest.parser import extract_images

pdf = (glob.glob("*.pdf") + glob.glob(str(DATA_DIR / "**/*.pdf"), recursive=True))[0]
figs = extract_images(pdf, DATA_DIR / "images")
print(f"抽出 {len(figs)} 张图表（已滤掉 logo 等小图）：")
for f in figs:
    print(f"  p{f['page']} x{f['xref']}  {f['width']}x{f['height']}  ->  {f['path']}")