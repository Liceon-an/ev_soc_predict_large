"""
Plot 09: Summary Dashboard
7-panel dashboard with summary table — one-page overview.
"""
import matplotlib.pyplot as plt
from plot_utils import OUTPUT_DIR, WINDOWS, W_LABELS, x, M, C, CORR_AP, LINR2


def plot_dashboard():
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: R² Trend
    ax1 = fig.add_subplot(gs[0, 0])
    r2 = [M[w]["r2"] for w in WINDOWS]
    ax1.plot(x, r2, "o-", color=C["r2"], lw=2.5, ms=10)
    for i, v in enumerate(r2):
        ax1.annotate(f"{v:.4f}", (x[i], v), (0, 10), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(W_LABELS, fontsize=8)
    ax1.set_title("R² Trend", fontsize=13, fontweight="bold")
    ax1.set_ylim(0.75, 0.97)

    # Panel 2: MAE + RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    mae = [M[w]["mae"] for w in WINDOWS]
    rmse = [M[w]["rmse"] for w in WINDOWS]
    ax2.plot(x, mae, "s-", color=C["mae"], lw=2, label="MAE")
    ax2.plot(x, rmse, "^-", color=C["rmse"], lw=2, label="RMSE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(W_LABELS, fontsize=8)
    ax2.set_title("MAE & RMSE", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel 3: MAPE
    ax3 = fig.add_subplot(gs[0, 2])
    mape = [M[w]["mape"] for w in WINDOWS]
    bars = ax3.bar(x, mape, color=C["mape"], alpha=0.7, width=50)
    for b, v in zip(bars, mape):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{v:.1f}%",
                 ha="center", fontsize=8, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(W_LABELS, fontsize=8)
    ax3.set_title("MAPE (%)", fontsize=13, fontweight="bold")

    # Panel 4: Normalized MAE
    ax4 = fig.add_subplot(gs[1, 0])
    vals = [M[w]["mae"] / M[w]["ys"] for w in WINDOWS]
    ax4.bar(x, vals, color=C["mae"], alpha=0.6, width=50)
    ax4.plot(x, vals, "o-", color="darkred", lw=2)
    for i, v in enumerate(vals):
        ax4.text(x[i], v + 0.003, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(W_LABELS, fontsize=8)
    ax4.set_title("Normalized MAE (MAE/σ)", fontsize=13, fontweight="bold")

    # Panel 5: LSTM vs Linear
    ax5 = fig.add_subplot(gs[1, 1])
    r2lin = [LINR2[w] for w in WINDOWS]
    wd = 35
    ax5.bar(x - wd / 2, r2lin, wd, label="Linear", color=C["linear"], alpha=0.6)
    ax5.bar(x + wd / 2, r2, wd, label="LSTM", color=C["lstm"], alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(W_LABELS, fontsize=8)
    ax5.legend(fontsize=8)
    ax5.set_title("LSTM vs Linear", fontsize=13, fontweight="bold")
    ax5.set_ylim(0.65, 0.97)

    # Panel 6: Correlation
    ax6 = fig.add_subplot(gs[1, 2])
    cv = [CORR_AP[w] for w in WINDOWS]
    ax6.plot(x, cv, "s-", color=C["corr"], lw=2.5, ms=8)
    for i, v in enumerate(cv):
        ax6.text(x[i], v + 0.004, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(W_LABELS, fontsize=8)
    ax6.set_title("avg_power vs ΔSoC corr", fontsize=12, fontweight="bold")
    ax6.set_ylim(0.80, 0.96)

    # Panel 7: Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    cols = ["Window"] + W_LABELS
    rows = ["R²", "MAE", "RMSE", "MAPE", "Samples", "ΔSoC mean", "ΔSoC std", "Coverage"]
    data = [[f"{M[w]['r2']:.4f}", f"{M[w]['mae']:.4f}", f"{M[w]['rmse']:.4f}",
             f"{M[w]['mape']:.1f}%", str(M[w]['n']), f"{M[w]['ym']:.4f}",
             f"{M[w]['ys']:.4f}", f"{M[w]['cov']}%"] for w in WINDOWS]
    cell = [[data[j][i] for j in range(len(WINDOWS))] for i in range(len(rows))]
    tbl = ax7.table(cellText=cell, rowLabels=rows, colLabels=cols,
                    cellLoc="center", loc="center", colWidths=[0.08] * 8)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cel in tbl.get_celld().items():
        if r == 0:
            cel.set_facecolor("#e3f2fd")
            cel.set_fontsize(10)
            cel.set_text_props(fontweight="bold")
        if c == 7:
            cel.set_facecolor("#e8f5e9")
    ax7.set_title("Complete Experimental Results", fontsize=14, fontweight="bold", pad=20)

    fig.suptitle("EV SoC Prediction — Ablation Study Dashboard", fontsize=18, fontweight="bold", y=0.98)
    fig.savefig(OUTPUT_DIR / "09_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("  [7/7] 09_dashboard.png")


if __name__ == "__main__":
    plot_dashboard()
