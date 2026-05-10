"""
Plot 17: Individual Feature Contribution — horizontal bar chart.
Top features shown separately; small contributors merged into "Others".
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

from src.analysis.feature_analysis import (
    run_phase2, load_test_data,
)

device = "cpu"
X_test, y_test, y_mean, y_std = load_test_data()
_, imp_df = run_phase2(X_test, y_test, y_mean, y_std, device)

all_features = imp_df["feature"].tolist()
all_imp = imp_df["importance_mean"].tolist()
all_std = imp_df["importance_std"].tolist()
total = sum(v for v in all_imp if v > 0)

# Features with >= 0.5% of total positive contribution stay separate
threshold = 0.005 * total

top_features = []
top_values = []
top_errors = []
others_sum = 0.0
others_std_sq = 0.0
others_count = 0

for f, v, s in zip(all_features, all_imp, all_std):
    if v >= threshold and v > 0:
        top_features.append(f)
        top_values.append(v)
        top_errors.append(s)
    else:
        if v > 0:
            others_sum += v
            others_std_sq += s ** 2
            others_count += 1

if others_count > 0:
    top_features.append(f"Others ({others_count} features)")
    top_values.append(others_sum)
    top_errors.append(float(np.sqrt(others_std_sq)))

# Reverse for horizontal bar (bottom to top)
top_features.reverse()
top_values.reverse()
top_errors.reverse()

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

n = len(top_features)
y_pos = range(n)

# Color gradient: top contributors darker
blues = plt.cm.Blues(np.linspace(0.3, 0.9, n))
# Make the last bar (Others) grey
blues[-1] = [0.69, 0.74, 0.78, 1.0]

bars = ax.barh(y_pos, top_values,
               color=blues, edgecolor="white", height=0.65)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features, fontsize=11)
ax.set_xlabel("R2 Drop (permuted)", fontsize=12)
ax.set_title("Individual Feature Permutation Importance (S1500, n=123)",
             fontsize=14, fontweight="bold")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.invert_yaxis()

# Value labels
for b, v in zip(bars, top_values):
    pct = v / total * 100
    ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
            f"{v:.4f}  ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")

# Total annotation
ax.text(0.98, 0.02, f"Baseline R2 = 0.9462  |  Total positive dR2 = {total:.4f}",
        transform=ax.transAxes, ha="right", fontsize=9, color="gray")

plt.tight_layout()
outdir = _PROJECT_ROOT / "plots"
outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "17_feature_bar.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  [17] 17_feature_bar.png saved")

# Print breakdown
print()
print("Feature Contribution Breakdown:")
for f, v in zip(all_features, all_imp):
    pct = v / total * 100 if v > 0 else 0
    flag = " <<<" if v >= threshold else ""
    print(f"  {f:35s}  dR2={v:+.4f}  {pct:5.1f}%{flag}")
