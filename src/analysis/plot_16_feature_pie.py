"""
Plot 16: 特征组贡献 — 饼图。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 中文字体 ──
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13})

from src.analysis.feature_analysis import run_phase3, load_test_data, load_model

device = "cpu"
X_test, y_test, y_mean, y_std = load_test_data()
model = load_model(device)
gdf = run_phase3(model, X_test, y_test, y_mean, y_std, device)

imp_vals = gdf["importance_mean"].tolist()
feat_names = gdf["features"].tolist()
total = sum(imp_vals)


def make_label(feats_str):
    s = str(feats_str)
    if "total_current" in s: return "电学特征"
    elif "temperature" in s: return "气象特征"
    elif "speed_diff_window20_mean" in s: return "速度（窗口统计）"
    elif "speed_window20_std" in s: return "速度（瞬时）"
    elif "cruising_ratio" in s: return "驾驶模式"
    elif "mileage_diff" in s: return "里程"
    return "其他"


clean = [make_label(f) for f in feat_names]
threshold_pct = 2.0
main_groups, main_vals = [], []
small_clean, small_vals = [], []

for g, v in zip(clean, imp_vals):
    if v / total * 100 >= threshold_pct:
        main_groups.append(g); main_vals.append(v)
    else:
        small_clean.append(g); small_vals.append(v)

main_groups.append("4 个小特征组")
main_vals.append(sum(small_vals))

fig = plt.figure(figsize=(20, 9))
ax1 = fig.add_subplot(1, 2, 1)
ax1.pie(main_vals, labels=main_groups, autopct="%1.1f%%",
        colors=["#1565C0", "#42A5F5", "#B0BEC5"],
        explode=[0.05, 0.03, 0.03], startangle=140,
        textprops={"fontsize": 13})
for at in ax1.texts[-len(main_groups):]:
    pass
for t in ax1.texts:
    t.set_fontsize(13)
ax1.set_title("特征组贡献度（R² 下降）", fontsize=15, fontweight="bold", pad=20)

ax2 = fig.add_subplot(1, 2, 2)
wedges2, _, _ = ax2.pie(
    small_vals, autopct="%1.1f%%",
    colors=["#64B5F6", "#90CAF9", "#BBDEFB", "#E3F2FD"],
    explode=[0.04, 0.04, 0.04, 0.04], startangle=90,
    textprops={"fontsize": 11}, pctdistance=0.58)
ax2.legend(wedges2,
           [f"{g}: {v:.4f}" for g, v in zip(small_clean, small_vals)],
           title="次要特征组", loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11, title_fontsize=12)
ax2.set_title("次要特征组展开", fontsize=15, fontweight="bold", pad=20)

fig.suptitle("EV SoC 模型 — 特征组贡献度（S1500, n=123）", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
outdir = _PROJECT_ROOT / "plots"
outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "16_feature_contribution_pie.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  [16] 已保存")
