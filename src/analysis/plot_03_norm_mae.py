"""
Plot 03: Normalized Error (MAE / ΔSoC_std)
Scale-invariant error metric across windows.
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def plot_norm_mae():
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = [M[w]["mae"] / M[w]["ys"] for w in WINDOWS]
    bars = ax.bar(x, vals, color=C["mae"], alpha=0.8, width=60, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.plot(x, vals, "o-", color="darkred", lw=2, ms=8, zorder=3)
    ax.set_xlabel("Window (seconds)")
    ax.set_ylabel("MAE / ΔSoC_std")
    ax.set_title("Normalized Error (scale-invariant)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(W_LABELS)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_normalized_mae.png", bbox_inches="tight")
    plt.close(fig)
    print("  [3/7] 03_normalized_mae.png")


if __name__ == "__main__":
    plot_norm_mae()
