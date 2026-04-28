"""
Plot 04: LSTM-Transformer vs Linear Baseline
Grouped bar chart with gain annotations.
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C, LINR2


def plot_lstm_vs_linear():
    fig, ax = plt.subplots(figsize=(12, 6))
    r2_lstm = [M[w]["r2"] for w in WINDOWS]
    r2_lin = [LINR2[w] for w in WINDOWS]
    gains = [r2_lstm[i] - r2_lin[i] for i in range(len(WINDOWS))]
    wd = 40
    ax.bar(x - wd / 2, r2_lin, wd, label="Linear (avg_power only)", color=C["linear"], alpha=0.7)
    ax.bar(x + wd / 2, r2_lstm, wd, label="LSTM-Transformer", color=C["lstm"], alpha=0.8)
    for i in range(len(WINDOWS)):
        ax.text(x[i] + wd / 2, r2_lstm[i] + 0.008, f"+{gains[i]:.3f}", ha="center",
                fontsize=8, fontweight="bold", color="darkgreen", rotation=90)
    ax.set_xlabel("Window (seconds)")
    ax.set_ylabel("R²")
    ax.set_title("LSTM-Transformer vs Linear Baseline (avg_power only)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(W_LABELS)
    ax.legend(loc="lower right")
    ax.set_ylim(0.65, 0.97)
    ax.text(0.98, 0.05, f"Avg gain: +{np.mean(gains):.3f} R²", transform=ax.transAxes,
            fontsize=12, ha="right", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_lstm_vs_linear.png", bbox_inches="tight")
    plt.close(fig)
    print("  [4/7] 04_lstm_vs_linear.png")


if __name__ == "__main__":
    plot_lstm_vs_linear()
