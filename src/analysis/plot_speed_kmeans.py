#!/usr/bin/env python3
"""
Plot speed KMeans clustering results.
Histogram colored by cluster with threshold lines + boxplot.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from speed_kmeans import run, OUTPUT_DIR

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "figure.figsize": (10, 6),
})

# Cluster colors (low/med/high)
CLUSTER_COLORS = ["#4CAF50", "#FF9800", "#F44336"]
CLUSTER_NAMES = ["Low", "Mid", "High"]


def plot_speed_histogram(result, output_path=None):
    """Plot speed histogram colored by KMeans clusters with threshold lines."""
    if output_path is None:
        output_path = OUTPUT_DIR / "11_speed_kmeans.png"

    speed = result["speed_nonzero"]
    labels = result["labels"]
    centers = result["centers"]
    thresholds = result["thresholds"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram bins (shared across all data for fair comparison)
    bins = np.linspace(speed.min(), speed.max(), 80)

    # Plot stacked histogram colored by cluster
    for cluster_id in range(3):
        mask = labels == cluster_id
        ax.hist(speed[mask], bins=bins, alpha=0.7,
                color=CLUSTER_COLORS[cluster_id],
                label=f"{CLUSTER_NAMES[cluster_id]} "
                      f"(center={centers[cluster_id]:.1f} km/h)",
                zorder=3)

    # Vertical lines for cluster centers
    for i, (name, center, color) in enumerate(
            zip(CLUSTER_NAMES, centers, CLUSTER_COLORS)):
        ax.axvline(center, color=color, linestyle="--", linewidth=2,
                   alpha=0.8, zorder=5)
        ax.text(center, ax.get_ylim()[1] * 0.92, f"  {name}\n  {center:.1f}",
                color=color, fontsize=9, fontweight="bold",
                va="top", ha="center")

    # Vertical lines for thresholds
    for t in thresholds:
        ax.axvline(t, color="#37474F", linestyle=":", linewidth=1.5,
                   alpha=0.7, zorder=4)

    # Annotate threshold regions
    ax.text((thresholds[0] + speed.min()) / 2, ax.get_ylim()[1] * 0.98,
            f"Threshold\n{thresholds[0]:.1f}",
            fontsize=8, ha="center", va="top", color="#37474F",
            fontweight="bold")
    ax.text((thresholds[0] + thresholds[1]) / 2, ax.get_ylim()[1] * 0.98,
            f"Threshold\n{thresholds[1]:.1f}",
            fontsize=8, ha="center", va="top", color="#37474F",
            fontweight="bold")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frequency")
    ax.set_title("Speed Distribution -- KMeans Clustering (Low / Mid / High)")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_speed_kmeans] Saved to {output_path}")


def plot_speed_boxplot(result, output_path=None):
    """Vertical boxplot per cluster showing separation."""
    if output_path is None:
        output_path = OUTPUT_DIR / "11_speed_kmeans_box.png"

    speed = result["speed_nonzero"]
    labels = result["labels"]

    fig, ax = plt.subplots(figsize=(7, 5))

    box_data = [speed[labels == i] for i in range(3)]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.4,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp["boxes"], CLUSTER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(CLUSTER_NAMES)
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Distribution by Cluster")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_speed_kmeans] Boxplot saved to {output_path}")


def main():
    result = run()
    plot_speed_histogram(result)
    plot_speed_boxplot(result)
    print("[plot_speed_kmeans] All plots done.")


if __name__ == "__main__":
    main()
