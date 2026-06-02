#!/usr/bin/env python3
"""
速度 KMeans 聚类结果可视化。
按聚类着色直方图（含阈值线）+ 箱线图。
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from speed_kmeans import run, OUTPUT_DIR

# --------------------------
# 先加载样式
# --------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "figure.figsize": (10, 6),
})

# ======================
# ✅ 【最后才设置字体】
# ✅ 放在所有配置之后
# ✅ 这才是真正生效的位置
# ======================
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 聚类颜色与中译名
CLUSTER_COLORS = ["#4CAF50", "#FF9800", "#F44336"]
CLUSTER_NAMES = ["低速", "中速", "高速"]


def plot_speed_histogram(result, output_path=None):
    if output_path is None:
        output_path = OUTPUT_DIR / "11_speed_kmeans.png"

    speed = result["speed_nonzero"]
    labels = result["labels"]
    centers = result["centers"]
    thresholds = result["thresholds"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(speed.min(), speed.max(), 80)

    for cluster_id in range(3):
        mask = labels == cluster_id
        ax.hist(speed[mask], bins=bins, alpha=0.7,
                color=CLUSTER_COLORS[cluster_id],
                label=f"{CLUSTER_NAMES[cluster_id]}（中心={centers[cluster_id]:.1f} km/h）",
                zorder=3)

    for i, (name, center, color) in enumerate(zip(CLUSTER_NAMES, centers, CLUSTER_COLORS)):
        ax.axvline(center, color=color, linestyle="--", linewidth=2, alpha=0.8, zorder=5)
        ax.text(center, ax.get_ylim()[1] * 0.92, f"  {name}\n  {center:.1f}",
                color=color, fontsize=16, fontweight="bold", va="top", ha="center")

    for t in thresholds:
        ax.axvline(t, color="#37474F", linestyle=":", linewidth=1.5, alpha=0.7, zorder=4)

    ax.text((thresholds[0] + speed.min()) / 2 + 10, ax.get_ylim()[1] * 0.98,
            f"分界\n{thresholds[0]:.1f}",
            fontsize=15, ha="center", va="top", color="#37474F", fontweight="bold")
    ax.text((thresholds[0] + thresholds[1]) / 2 + 10, ax.get_ylim()[1] * 0.98,
            f"分界\n{thresholds[1]:.1f}",
            fontsize=15, ha="center", va="top", color="#37474F", fontweight="bold")

    ax.set_xlabel("速度（km/h）", fontsize=20)
    ax.set_ylabel("频次", fontsize=20)
    ax.set_title("速度分布 — KMeans 聚类（低 / 中 / 高）", fontsize=24, fontweight="bold")
    ax.legend(fontsize=20, loc="upper right")
    ax.tick_params(labelsize=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {output_path}")


def plot_speed_boxplot(result, output_path=None):
    if output_path is None:
        output_path = OUTPUT_DIR / "11_speed_kmeans_box.png"

    speed = result["speed_nonzero"]
    labels = result["labels"]

    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [speed[labels == i] for i in range(3)]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.4,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp["boxes"], CLUSTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(CLUSTER_NAMES, fontsize=20)
    ax.set_ylabel("速度（km/h）", fontsize=20)
    ax.set_title("各聚类速度分布", fontsize=24, fontweight="bold")
    ax.tick_params(labelsize=20)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"箱线图已保存 {output_path}")


def main():
    result = run()
    plot_speed_histogram(result)
    plot_speed_boxplot(result)
    print("全部完成。")


if __name__ == "__main__":
    main()