"""
Plot 01: R² vs Window Length
Shows R² trend across 200s-1500s with inflection annotations.
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C


def plot_r2():
    fig, ax = plt.subplots(figsize=(10, 6))
    r2 = [M[w]["r2"] for w in WINDOWS]
    ax.plot(x, r2, "o-", color=C["r2"], lw=2.5, ms=10, zorder=3)
    ax.fill_between(x, 0.77, r2, alpha=0.1, color=C["r2"])
    for i, v in enumerate(r2):
        off = 10 if WINDOWS[i] != 1500 else -25
        ax.annotate(f"{v:.4f}", (x[i], v), (0, off), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color=C["r2"])
    ax.axvspan(200, 400, alpha=0.06, color="blue", label="Signal-dominated")
    ax.axvspan(400, 800, alpha=0.06, color="green", label="Context-dominated")
    ax.axvspan(800, 1200, alpha=0.06, color="orange", label="Diminishing returns")
    ax.axvspan(1200, 1500, alpha=0.06, color="red", label="2nd jump")
    ax.set_xlabel("Window (seconds)")
    ax.set_ylabel("R²")
    ax.set_title("R² vs Window Length", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(W_LABELS)
    ax.set_ylim(0.75, 0.97)
    ax.legend(loc="lower right", fontsize=9, title="Dominant factor", title_fontsize=9)
    ax.annotate("S800 inflection\nR²=0.9157", (800, 0.9157), (620, 0.89),
                fontsize=10, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))
    ax.annotate("S1500 best\nR²=0.9462", (1500, 0.9462), (1220, 0.93),
                fontsize=10, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_r2_trend.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1/7] 01_r2_trend.png")


if __name__ == "__main__":
    plot_r2()
