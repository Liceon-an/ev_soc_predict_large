"""
Plot 02: 全部指标对比 — 拆分为 3 张子图。
  - 02a: R² + MAPE
  - 02b: MAE + RMSE（原始）
  - 02c: 归一化 MAE + 归一化 RMSE（÷ΔSoC 标准差）
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def _fmt(val, key):
    if key == "mape":
        return f"{val:.2f}%"
    elif abs(val) >= 10:
        return f"{val:.2f}"
    elif abs(val) >= 1:
        return f"{val:.3f}"
    else:
        return f"{val:.4f}"


def _plot_pair(items, filename, suptitle):
    """1x2 柱状图。items = [(vals, name, key, color), (vals, name, key, color)]"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (vals, name, key, color) in zip(axes, items):
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=60, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v, key),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xlabel("窗口时长", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(W_LABELS, fontsize=11)
        ax.set_title(f"测试集 {name}", fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=10)
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_metrics():
    # 图 a: R² + MAPE
    _plot_pair(
        [
            ([M[w]["r2"] for w in WINDOWS], "R²", "r2", C["r2"]),
            ([M[w]["mape"] for w in WINDOWS], "MAPE (%)", "mape", C["mape"]),
        ],
        "02a_r2_mape.png",
        "R² 与 MAPE 对比",
    )
    print("  [2a] 02a_r2_mape.png")

    # 图 b: MAE + RMSE（原始）
    _plot_pair(
        [
            ([M[w]["mae"] for w in WINDOWS], "MAE", "mae", C["mae"]),
            ([M[w]["rmse"] for w in WINDOWS], "RMSE", "rmse", C["rmse"]),
        ],
        "02b_mae_rmse.png",
        "MAE 与 RMSE 对比",
    )
    print("  [2b] 02b_mae_rmse.png")

    # 图 c: 归一化 MAE + 归一化 RMSE（÷ΔSoC 标准差）
    _plot_pair(
        [
            ([M[w]["mae"] / M[w]["ys"] for w in WINDOWS],
             "归一化 MAE", "norm_mae", C["mae"]),
            ([M[w]["rmse"] / M[w]["ys"] for w in WINDOWS],
             "归一化 RMSE", "norm_rmse", C["rmse"]),
        ],
        "02c_norm_mae_rmse.png",
        "归一化 MAE 与 RMSE（÷ΔSoC 标准差）",
    )
    print("  [2c] 02c_norm_mae_rmse.png")


if __name__ == "__main__":
    plot_metrics()
