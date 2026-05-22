"""
Plot 17: 单特征贡献度 — 水平柱状图。
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

from src.analysis.feature_analysis import run_phase2, load_test_data

device = "cpu"
X_test, y_test, y_mean, y_std = load_test_data()
_, imp_df = run_phase2(X_test, y_test, y_mean, y_std, device)

all_features = imp_df["feature"].tolist()
all_imp = imp_df["importance_mean"].tolist()
all_std = imp_df["importance_std"].tolist()
total = sum(v for v in all_imp if v > 0)
threshold = 0.005 * total

top_features, top_values, top_errors = [], [], []
others_sum, others_std_sq, others_count = 0.0, 0.0, 0

for f, v, s in zip(all_features, all_imp, all_std):
    if v >= threshold and v > 0:
        top_features.append(f); top_values.append(v); top_errors.append(s)
    elif v > 0:
        others_sum += v; others_std_sq += s**2; others_count += 1

if others_count > 0:
    top_features.append(f"其他（{others_count} 个特征）")
    top_values.append(others_sum)
    top_errors.append(float(np.sqrt(others_std_sq)))

top_features.reverse(); top_values.reverse(); top_errors.reverse()

fig, ax = plt.subplots(figsize=(14, 9))
n = len(top_features)
blues = plt.cm.Blues(np.linspace(0.3, 0.9, n))
blues[-1] = [0.69, 0.74, 0.78, 1.0]

bars = ax.barh(range(n), top_values, color=blues, edgecolor="white", height=0.65)
ax.set_yticks(range(n))
ax.set_yticklabels(top_features, fontsize=12)
ax.set_xlabel("R² 下降（特征被随机打乱后）", fontsize=13)
ax.set_title("单特征排列重要性（S1500, n=123）", fontsize=15, fontweight="bold")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.invert_yaxis()
ax.tick_params(labelsize=10)

for b, v in zip(bars, top_values):
    ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
            f"{v:.4f}  ({v/total*100:.1f}%)", va="center", fontsize=10, fontweight="bold")

ax.text(0.98, 0.02, f"基线 R² = 0.9462  |  总正向 ΔR² = {total:.4f}",
        transform=ax.transAxes, ha="right", fontsize=10, color="gray")

plt.tight_layout()
outdir = _PROJECT_ROOT / "plots"; outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "17_feature_bar.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  [17] 已保存")

print("\n特征贡献度明细:")
for f, v in zip(all_features, all_imp):
    pct = v / total * 100 if v > 0 else 0
    flag = " <<<" if v >= threshold else ""
    print(f"  {f:35s}  ΔR²={v:+.4f}  {pct:5.1f}%{flag}")
