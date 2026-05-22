"""
Plot 04: LSTM-Transformer vs 线性基线
分组柱状图 + 增益标注。
"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C, LINR2


def plot_lstm_vs_linear():
    fig, ax = plt.subplots(figsize=(14, 7))
    r2_lstm = [M[w]["r2"] for w in WINDOWS]
    r2_lin = [LINR2[w] for w in WINDOWS]
    gains = [r2_lstm[i] - r2_lin[i] for i in range(len(WINDOWS))]
    wd = 40
    ax.bar(x - wd / 2, r2_lin, wd, label="线性模型（仅 avg_power）", color=C["linear"], alpha=0.7)
    ax.bar(x + wd / 2, r2_lstm, wd, label="LSTM-Transformer", color=C["lstm"], alpha=0.8)
    for i in range(len(WINDOWS)):
        ax.text(x[i] + wd / 2, r2_lstm[i] + 0.008, f"+{gains[i]:.3f}", ha="center",
                fontsize=9, fontweight="bold", color="darkgreen", rotation=90)
    ax.set_xlabel("窗口时长（秒）", fontsize=13)
    ax.set_ylabel("R²", fontsize=13)
    ax.set_title("LSTM-Transformer vs 线性基线（仅 avg_power）", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(W_LABELS, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(0.65, 0.97)
    ax.text(0.98, 0.05, f"平均增益: +{np.mean(gains):.3f} R²", transform=ax.transAxes,
            fontsize=13, ha="right",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_lstm_vs_linear.png", bbox_inches="tight")
    plt.close(fig)
    print("  [4] 04_lstm_vs_linear.png")


if __name__ == "__main__":
    plot_lstm_vs_linear()
