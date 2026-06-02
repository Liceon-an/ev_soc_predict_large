#!/usr/bin/env python3
"""
原始SOC vs 重构SOC 连续变化对比图

从 aligned_data_refined_soc.csv 中提取目标行程，
对比原始SOC (standard_soc, 整数离散) 与重构SOC (refined_soc, 安时积分平滑) 的变化。

用法:
  python3 src/analysis/plot_soc_comparison.py           # 默认 Trip 1166
  python3 src/analysis/plot_soc_comparison.py --trip 1484

输出:
  plots/soc_comparison_trip{id}.png
"""

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 路径配置 ────────────────────────────────────────────
PROJECT_ROOT = Path("/root/code/ev_soc_predict")
DATA_PATH = PROJECT_ROOT / "data" / "aligned" / "aligned_data_refined_soc.csv"
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 中文字体 ────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 行程选择 ────────────────────────────────────────────
CANDIDATE_TRIPS = {
    1166: {"color": "#1565C0"},
    1159: {"color": "#2E7D32"},
    1484: {"color": "#C62828"},
    1643: {"color": "#6A1B9A"},
}
DEFAULT_TRIP = 1166


def load_trip_data(trip_id):
    df = pd.read_csv(DATA_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
    trip_df = df[df["trip_id"] == trip_id].copy().reset_index(drop=True)
    if trip_df.empty:
        raise ValueError(f"Trip {trip_id} 不存在")
    return trip_df


def plot_comparison(trip_df, trip_id):
    t0 = trip_df["DATE"].iloc[0]
    elapsed_min = (trip_df["DATE"] - t0).dt.total_seconds().values / 60.0

    refined = trip_df["refined_soc"].values
    standard = trip_df["standard_soc"].values

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(elapsed_min, refined, color="#1565C0", linewidth=2.5, alpha=0.9,
            label="重构SOC", zorder=4)
    ax.plot(elapsed_min, standard, color="#FF6F00", linewidth=2.0, alpha=0.85,
            drawstyle="steps-post", label="原始SOC", zorder=3)

    ax.fill_between(elapsed_min, refined, standard, alpha=0.08, color="#7B1FA2")

    ax.set_xlabel("经过时间 (分钟)", fontsize=18)
    ax.set_ylabel("SOC (%)", fontsize=18)
    ax.set_title("原始SOC与重构SOC连续变化对比", fontsize=22, fontweight="bold")
    ax.legend(loc="lower left", fontsize=16, framealpha=0.9)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"soc_comparison_trip{trip_id}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    max_diff = np.abs(refined - standard).max()
    mean_diff = np.abs(refined - standard).mean()
    print(f"  {out_path.name}  时长={elapsed_min[-1]:.1f}min  "
          f"SoC {refined[0]:.1f}%→{refined[-1]:.1f}%  "
          f"|diff|_max={max_diff:.2f}%  |diff|_mean={mean_diff:.3f}%")


def main(trip_id=DEFAULT_TRIP):
    print(f"[plot_soc_comparison] Trip {trip_id}")
    trip_df = load_trip_data(trip_id)
    plot_comparison(trip_df, trip_id)
    print("  完成.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="原始SOC vs 重构SOC 对比图")
    parser.add_argument("--trip", type=int, default=DEFAULT_TRIP,
                        choices=list(CANDIDATE_TRIPS.keys()),
                        help=f"选择行程 ID, 可选: {list(CANDIDATE_TRIPS.keys())}")
    args = parser.parse_args()
    main(args.trip)
