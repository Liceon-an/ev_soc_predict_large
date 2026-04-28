"""
Plot 10: Normalized Metrics Comparison
Replace MAE/RMSE with normalized-by-ΔSoC_std versions for fair comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def plot_norm_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    items = [
        ("R²", "r2", C["r2"], False),
        ("MAE / ΔSoC_std", "mae", C["mae"], True),
        ("RMSE / ΔSoC_std", "rmse", C["rmse"], True),
        ("MAPE (%)", "mape", C["mape"], False),
    ]
    for idx, (name, key, color, do_norm) in enumerate(items):
        ax = axes[idx // 2][idx % 2]
        if do_norm:
            vals = [M[w][key] / M[w]["ys"] for w in WINDOWS]
        else:
            vals = [M[w][key] for w in WINDOWS]
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=60, edgecolor="white")
        for b, v in zip(bars, vals):
            if key == "mape":
                label = f"{v:.2f}%"
            elif do_norm:
                label = f"{v:.3f}"
            else:
                label = f"{v:.4f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_xticks(x)
        ax.set_xticklabels(W_LABELS, fontsize=9)
        ax.set_title(f"Test {name}", fontsize=13, fontweight="bold")
    fig.suptitle("Normalized Metrics Comparison (MAE/σ, RMSE/σ)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_normalized_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1/1] 10_normalized_metrics.png")


if __name__ == "__main__":
    plot_norm_metrics()
