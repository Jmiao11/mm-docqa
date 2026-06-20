"""统一的路径锚点：所有数据目录基于项目根的绝对路径，不受启动 cwd 影响。"""
from pathlib import Path

# 本文件在 core/ 下，上一级就是项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
GOLDEN_PATH = DATA_DIR / "golden.jsonl"