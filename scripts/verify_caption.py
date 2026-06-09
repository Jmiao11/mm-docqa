"""真调 Moonshot 给 2 张图生成 caption，肉眼检验质量。python scripts/verify_caption.py"""
import glob
from pathlib import Path
from core.paths import DATA_DIR
from ingest.captioner import Captioner

from dotenv import load_dotenv
load_dotenv()   # ← 加在所有 import 之后、其他代码之前

img_dir = DATA_DIR / "images"
targets = (glob.glob(str(img_dir / "*_p7_x55.png"))      # 社会网络图
           + glob.glob(str(img_dir / "*_p4_x64.png")))   # p4 的一张图

cap = Captioner()   # 默认读 MOONSHOT_API_KEY、base_url=api.moonshot.cn/v1、model=kimi-k2.6
for png in targets:
    print("=" * 60); print(Path(png).name)
    print(cap.caption(png))