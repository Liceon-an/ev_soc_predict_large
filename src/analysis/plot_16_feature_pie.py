"""
Plot 16: Feature Group Contribution — pie chart.
Main groups shown in left pie; minor groups expanded in right sub-pie.
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
    run_phase3, load_test_data, load_model,
)

device = "cpu"
X_test, y_test, y_mean, y_std = load_test_data()
model = load_model(device)

gdf = run_phase3(model, X_test, y_test, y_mean, y_std, device)

imp_vals = gdf["importance_mean"].tolist()
feat_names = gdf["features"].tolist()
total = sum(imp_vals)


def make_label(feats):
    if "total_current" in feats:
        return "Electrical"
    elif "temperature" in feats:
        return "Weather"
    elif "speed_diff_window20_mean" in feats:
        return "Speed (window)"
    elif "speed_window20_std" in feats:
        return "Speed (raw)"
    elif "cruising_ratio" in feats:
        return "Driving Mode"
    elif "mileage_diff" in feats:
        return "Mileage"
    return "Other"


clean = [make_label(f) for f in feat_names]

threshold_pct = 2.0
main_groups, main_vals = [], []
small_clean, small_vals = [], []

for g, v in zip(clean, imp_vals):
    pct = v / total * 100
    if pct >= threshold_pct:
        main_groups.append(g)
        main_vals.append(v)
    else:
        small_clean.append(g)
        small_vals.append(v)

main_groups.append("4 Minor Groups")
main_vals.append(sum(small_vals))

# --- Figure ---
fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(1, 2, 1)
wedges, texts, autotexts = ax1.pie(
    main_vals, labels=main_groups, autopct="%1.1f%%",
    colors=["#1565C0", "#42A5F5", "#B0BEC5"],
    explode=[0.05, 0.03, 0.03],
    startangle=140,
    textprops={"fontsize": 12},
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(10)
ax1.set_title("Feature Group Contribution (R2 Drop)", fontsize=14, fontweight="bold", pad=20)

ax2 = fig.add_subplot(1, 2, 2)
wedges2, texts2, autotexts2 = ax2.pie(
    small_vals, autopct="%1.1f%%",
    colors=["#64B5F6", "#90CAF9", "#BBDEFB", "#E3F2FD"],
    explode=[0.04, 0.04, 0.04, 0.04],
    startangle=90,
    textprops={"fontsize": 10},
    pctdistance=0.58,
)
for at in autotexts2:
    at.set_fontweight("bold")
    at.set_fontsize(9)

ax2.legend(
    wedges2,
    [f"{g}: {v:.4f}" for g, v in zip(small_clean, small_vals)],
    title="Minor Groups",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10,
    title_fontsize=11,
)
ax2.set_title("Minor Groups Breakdown", fontsize=14, fontweight="bold", pad=20)

fig.suptitle("EV SoC Model -- Feature Group Contribution (S1500, n=123)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
outdir = _PROJECT_ROOT / "plots"
outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "16_feature_contribution_pie.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  [16] 16_feature_contribution_pie.png saved")
