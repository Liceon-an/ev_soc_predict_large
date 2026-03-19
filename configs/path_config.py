import os
from pathlib import Path

# 获取项目根目录 (ev_soc_predict/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ALIGNED_DATA_DIR = DATA_DIR / "aligned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 源码目录
SRC_DIR = ROOT_DIR / "src"

# 结果目录
RESULT_DIR = ROOT_DIR / "result"
MODEL_DIR = ROOT_DIR / "models"
PLOT_DIR = ROOT_DIR / "plot"

def get_path(key: str) -> Path:
    """简单的路径获取助手，可根据需要扩展"""
    paths = {
        "root": ROOT_DIR,
        "processed": PROCESSED_DATA_DIR,
        "aligned": ALIGNED_DATA_DIR,
        "raw": RAW_DATA_DIR,
        "config": ROOT_DIR / "configs" / "config.yaml"
    }
    return paths.get(key, ROOT_DIR)

# 确保目录存在
for d in [PROCESSED_DATA_DIR, RESULT_DIR, MODEL_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)