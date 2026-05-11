"""
Plot 02: All Metrics Comparison — split into 3 separate files.
  - 02a: R² + MAPE
  - 02b: MAE + RMSE (original)
  - 02c: Normalized MAE + Normalized RMSE (÷ΔSoC std)
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
    """1x2 bar chart. items = [(vals, name, key, color), (vals, name, key, color)]"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (vals, name, key, color) in zip(axes, items):
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=60, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v, key),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_xticks(x)
        ax.set_xticklabels(W_LABELS, fontsize=9)
        ax.set_title(f"Test {name}", fontsize=13, fontweight="bold")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_metrics():
    # ---- File a: R2 + MAPE ----
    _plot_pair(
        [
            ([M[w]["r2"] for w in WINDOWS], "R²", "r2", C["r2"]),
            ([M[w]["mape"] for w in WINDOWS], "MAPE (%)", "mape", C["mape"]),
        ],
        "02a_r2_mape.png",
        "R² and MAPE Comparison",
    )
    print("  [2a] 02a_r2_mape.png")

    # ---- File b: MAE + RMSE (original) ----
    _plot_pair(
        [
            ([M[w]["mae"] for w in WINDOWS], "MAE", "mae", C["mae"]),
            ([M[w]["rmse"] for w in WINDOWS], "RMSE", "rmse", C["rmse"]),
        ],
        "02b_mae_rmse.png",
        "MAE and RMSE Comparison",
    )
    print("  [2b] 02b_mae_rmse.png")

    # ---- File c: Normalized MAE + Normalized RMSE (/Delta SoC std) ----
    _plot_pair(
        [
            ([M[w]["mae"] / M[w]["ys"] for w in WINDOWS],
             "Normalized MAE", "norm_mae", C["mae"]),
            ([M[w]["rmse"] / M[w]["ys"] for w in WINDOWS],
             "Normalized RMSE", "norm_rmse", C["rmse"]),
        ],
        "02c_norm_mae_rmse.png",
        "Normalized MAE and RMSE (÷ΔSoC std)",
    )
    print("  [2c] 02c_norm_mae_rmse.png")


if __name__ == "__main__":
    plot_metrics()
