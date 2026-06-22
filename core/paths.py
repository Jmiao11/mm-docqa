"""统一的路径锚点：所有数据目录基于项目根的绝对路径，不受启动 cwd 影响。"""
from pathlib import Path

# 本文件在 core/ 下，上一级就是项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
# 黄金集是手写、需版本化的【源数据】，性质与 data/ 下的运行时生成物
# (chroma/uploads/sqlite) 相反，且 data/ 被 .gitignore 整目录忽略。故锚定到
# evaluators/ 旁(与读它的 loader 同处)，不放 data/，保持「data/ 一律不提交」纯净。
GOLDEN_PATH = PROJECT_ROOT / "evaluators" / "golden.jsonl"