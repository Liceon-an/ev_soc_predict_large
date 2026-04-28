"""
Plot 08: Sample Count vs R² Trade-off
Dual-axis: train sample bars + R² line + coverage annotations.
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def plot_tradeoff():
    fig, ax1 = plt.subplots(figsize=(12, 6))
    n = [M[w]["n"] for w in WINDOWS]
    r2 = [M[w]["r2"] for w in WINDOWS]
    cov = [M[w]["cov"] for w in WINDOWS]

    bars = ax1.bar(x, n, alpha=0.3, color=C["samples"], width=50,
                   edgecolor="white", label="Train samples")
    ax1.set_ylabel("Train samples", color=C["samples"])
    ax1.tick_params(axis="y", labelcolor=C["samples"])
    for b, s in zip(bars, n):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{s}",
                 ha="center", fontsize=9, color=C["samples"], fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, r2, "o-", color=C["r2"], lw=2.5, ms=10, label="R²", zorder=3)
    ax2.set_ylabel("R²", color=C["r2"])
    ax2.tick_params(axis="y", labelcolor=C["r2"])
    ax2.set_ylim(0.70, 0.97)

    for i, (w, c) in enumerate(zip(WINDOWS, cov)):
        ax2.annotate(f"Coverage\n{c}%", (x[i], r2[i]), (0, -50),
                     textcoords="offset points", ha="center", fontsize=8, color="gray")

    ax1.set_xlabel("Window (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(W_LABELS)
    ax1.set_title("Sample Count vs R² Trade-off", fontsize=14, fontweight="bold")
    l1, l2 = ax1.get_legend_handles_labels()
    l3, l4 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l3, l2 + l4, loc="center right", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_sample_size_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    print("  [6/7] 08_sample_size_tradeoff.png")


if __name__ == "__main__":
    plot_tradeoff()
