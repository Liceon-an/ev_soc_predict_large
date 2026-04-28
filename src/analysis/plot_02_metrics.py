"""
Plot 02: All Metrics Comparison
2x2 bar charts: R², MAE, RMSE, MAPE across windows.
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def plot_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    items = [
        ("R²", "r2", C["r2"]),
        ("MAE", "mae", C["mae"]),
        ("RMSE", "rmse", C["rmse"]),
        ("MAPE (%)", "mape", C["mape"]),
    ]
    for idx, (name, key, color) in enumerate(items):
        ax = axes[idx // 2][idx % 2]
        vals = [M[w][key] for w in WINDOWS]
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=60, edgecolor="white")
        for b, v in zip(bars, vals):
            label = f"{v:.4f}" if key != "mape" else f"{v:.2f}%"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_xticks(x)
        ax.set_xticklabels(W_LABELS, fontsize=9)
        ax.set_title(f"Test {name}", fontsize=13, fontweight="bold")
    fig.suptitle("All Metrics Comparison", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [2/7] 02_metrics_comparison.png")


if __name__ == "__main__":
    plot_metrics()
