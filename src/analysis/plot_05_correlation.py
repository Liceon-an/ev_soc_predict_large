"""
Plot 05: Feature-Label Correlation & LSTM R²
Dual-axis: avg_power vs ΔSoC correlation (line) + LSTM R² (bars).
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C, CORR_AP


def plot_corr():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    cvals = [CORR_AP[w] for w in WINDOWS]
    r2 = [M[w]["r2"] for w in WINDOWS]
    ax1.plot(x, cvals, "s-", color=C["corr"], lw=2.5, ms=10, label="avg_power vs ΔSoC corr")
    ax1.set_ylabel("Correlation", color=C["corr"])
    ax1.tick_params(axis="y", labelcolor=C["corr"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(W_LABELS)
    ax1.set_ylim(0.80, 0.96)
    for i, v in enumerate(cvals):
        ax1.annotate(f"{v:.4f}", (x[i], v), (0, 12), textcoords="offset points",
                     ha="center", fontsize=9, color=C["corr"])
    ax2 = ax1.twinx()
    ax2.bar(x, r2, alpha=0.15, color=C["lstm"], width=50, label="LSTM R²")
    ax2.set_ylabel("LSTM R²", color=C["lstm"])
    ax2.tick_params(axis="y", labelcolor=C["lstm"])
    ax2.set_ylim(0.70, 0.97)
    l1, l2 = ax1.get_legend_handles_labels()
    l3, l4 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l3, l2 + l4, loc="lower right", fontsize=10)
    ax1.set_title("Feature-Label Correlation & LSTM R² (Physical Signal Strengthens)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_correlation_trend.png", bbox_inches="tight")
    plt.close(fig)
    print("  [5/7] 05_correlation_trend.png")


if __name__ == "__main__":
    plot_corr()
