#!/usr/bin/env python3
"""
Plot 06: Predicted vs True ΔSoC Scatter — 7 models, per-window inference.
Layout: row 1 = 4 subplots, row 2 = 3 subplots centered.
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_utils import OUTPUT_DIR, WINDOWS
from configs.model_config import model_config
from src.models.lstm_transformer import LSTMTransformer

Y_STATS = {
    200:  (0.3405, 0.3377),
    400:  (0.6788, 0.5918),
    600:  (0.9879, 0.8202),
    800:  (1.2873, 1.0277),
    1000: (1.5389, 1.1982),
    1200: (1.8055, 1.3674),
    1500: (2.1699, 1.6087),
}


def load_model(checkpoint_path, device):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = LSTMTransformer(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(model, device, data_dir, ymean, ystd):
    test_path = Path(data_dir) / "test_data.npz"
    data = np.load(str(test_path))
    X = torch.from_numpy(data["X"]).float()
    y_true_raw = data["y_main"]

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch_x = X[i:i + 64].to(device)
            batch_pred_norm, _ = model(batch_x)
            batch_pred_raw = batch_pred_norm.cpu().numpy() * ystd + ymean
            preds.append(batch_pred_raw.flatten() if batch_pred_raw.ndim > 1 else batch_pred_raw)

    y_pred_raw = np.concatenate(preds)
    return y_true_raw, y_pred_raw


def plot_scatter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[plot_06] Device: {device}")

    fig = plt.figure(figsize=(28, 16))

    # Row 1: 4 columns (S200 S400 S600 S800)
    # Row 2: 3 columns (S1000 S1200 S1500), centered with offset
    gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.35, wspace=0.25)

    # Row 1: each subplot spans 3 of 12 columns
    ax_positions = [
        (0, slice(0, 3)),     # 200s
        (0, slice(3, 6)),     # 400s
        (0, slice(6, 9)),     # 600s
        (0, slice(9, 12)),    # 800s
    ]
    # Row 2: each subplot spans 3 of 12 columns, offset by 1.5 to center
    ax_positions += [
        (1, slice(1, 4)),     # 1000s  (offset 1, span 3)
        (1, slice(4, 7)),     # 1200s  (offset 4, span 3)
        (1, slice(7, 10)),    # 1500s  (offset 7, span 3)
    ]

    for idx, w in enumerate(WINDOWS):
        row, col_slice = ax_positions[idx]
        ax = fig.add_subplot(gs[row, col_slice])

        data_dir = _PROJECT_ROOT / "data" / "train" / f"split_{w}s"
        ymean, ystd = Y_STATS[w]
        ckpt_path = _PROJECT_ROOT / "models" / f"best_model_{w}s.pt"

        print(f"[plot_06]   {w}s  loading {ckpt_path.name} ...")
        model = load_model(ckpt_path, device)
        y_true, y_pred = run_inference(model, device, data_dir, ymean, ystd)

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap="Blues", mincnt=1)

        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        margin = max((vmax - vmin) * 0.08, 0.1)
        lo = max(vmin - margin, -0.5)
        hi = vmax + margin

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True ΔSoC")
        ax.set_ylabel("Predicted ΔSoC")
        ax.set_title(f"{w}s  R²={r2:.4f}  MAE={mae:.4f}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

        del model
        torch.cuda.empty_cache()

    fig.suptitle("Predicted vs True ΔSoC (7 models)", fontsize=18, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "06_scatter_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[plot_06]  Done → plots/06_scatter_comparison.png")


if __name__ == "__main__":
    plot_scatter()
