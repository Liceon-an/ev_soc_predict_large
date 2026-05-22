"""
Shared constants and data for all ablation study plots.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ── 中文字体设置 ──
fm._load_fontmanager(try_read_cache=False)
_font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
fm.fontManager.addfont(_font_path)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("/root/code/ev_soc_predict/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [200, 400, 600, 800, 1000, 1200, 1500]
W_LABELS = [f"{w}s" for w in WINDOWS]
x = np.array(WINDOWS)

M = {
    200:  {"r2": 0.7744, "mae": 0.0864, "rmse": 0.1665, "mape": 20.73, "n": 7782,  "ym": 0.3405, "ys": 0.3377, "cov": 94.4},
    400:  {"r2": 0.8285, "mae": 0.1327, "rmse": 0.2501, "mape": 21.36, "n": 3496,  "ym": 0.6788, "ys": 0.5918, "cov": 86.7},
    600:  {"r2": 0.8613, "mae": 0.1716, "rmse": 0.3122, "mape": 20.81, "n": 2117,  "ym": 0.9879, "ys": 0.8202, "cov": 83.9},
    800:  {"r2": 0.9157, "mae": 0.1860, "rmse": 0.2837, "mape": 16.97, "n": 1453,  "ym": 1.2873, "ys": 1.0277, "cov": 79.5},
    1000: {"r2": 0.9187, "mae": 0.2255, "rmse": 0.3316, "mape": 17.13, "n": 1066,  "ym": 1.5389, "ys": 1.1982, "cov": 75.0},
    1200: {"r2": 0.9227, "mae": 0.2526, "rmse": 0.3904, "mape": 18.24, "n": 812,   "ym": 1.8055, "ys": 1.3674, "cov": 70.3},
    1500: {"r2": 0.9462, "mae": 0.2343, "rmse": 0.3425, "mape": 21.81, "n": 574,   "ym": 2.1699, "ys": 1.6087, "cov": 64.4},
}

CORR_AP = {200: 0.8361, 400: 0.8901, 600: 0.9122, 800: 0.9282, 1000: 0.9361, 1200: 0.9405, 1500: 0.9429}
LINR2   = {200: 0.7182, 400: 0.7932, 600: 0.8515, 800: 0.8421, 1000: 0.8719, 1200: 0.8850, 1500: 0.9055}

C = {
    "r2": "#2196F3", "mae": "#FF5722", "rmse": "#9C27B0",
    "mape": "#FF9800", "linear": "#607D8B", "lstm": "#4CAF50",
    "corr": "#E91E63", "samples": "#00BCD4",
}
