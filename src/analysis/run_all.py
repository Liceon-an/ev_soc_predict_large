#!/usr/bin/env python3
"""
Run all active ablation study visualization plots.
Excludes: 06_scatter, 07_residuals (require GPU inference).
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure imports work from this directory
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "figure.figsize": (10, 6),
})

from plot_01_r2 import plot_r2
from plot_02_metrics import plot_metrics
from plot_04_lstm_vs_linear import plot_lstm_vs_linear
from plot_05_correlation import plot_corr


def main():
    print("=" * 50)
    print("  EV SoC Ablation — Visualization")
    print("=" * 50)

    # Static plots (no GPU needed)
    plot_r2()
    plot_metrics()
    plot_lstm_vs_linear()
    plot_corr()

    print("=" * 50)
    print("  All done! Output: plots/*.png")
    print("=" * 50)
    print("  Note: 06_scatter / 07_residuals excluded (need GPU inference)")
    print("=" * 50)


if __name__ == "__main__":
    main()
