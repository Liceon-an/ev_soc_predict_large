#!/usr/bin/env python3
"""
Plot 06: 预测值 vs 真实 ΔSoC 散点图 — 7 个模型，逐窗口推理。
布局: 上一行 4 列，下一行 3 列居中。
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

# 中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
    print(f"[plot_06] 设备: {device}")

    fig = plt.figure(figsize=(30, 17))
    gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.40, wspace=0.25)

    ax_positions = [
        (0, slice(0, 3)),
        (0, slice(3, 6)),
        (0, slice(6, 9)),
        (0, slice(9, 12)),
    ]
    ax_positions += [
        (1, slice(1, 4)),
        (1, slice(4, 7)),
        (1, slice(7, 10)),
    ]

    for idx, w in enumerate(WINDOWS):
        row, col_slice = ax_positions[idx]
        ax = fig.add_subplot(gs[row, col_slice])

        data_dir = _PROJECT_ROOT / "data" / "train" / f"split_{w}s"
        ymean, ystd = Y_STATS[w]
        ckpt_path = _PROJECT_ROOT / "models" / f"best_model_{w}s.pt"

        print(f"[plot_06]   窗口 {w}s  加载 {ckpt_path.name} ...")
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
        ax.set_xlabel("真实 ΔSoC", fontsize=13)
        ax.set_ylabel("预测 ΔSoC", fontsize=13)
        ax.set_title(f"窗口 {w}s   R²={r2:.4f}  MAE={mae:.4f}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

        del model
        torch.cuda.empty_cache()

    fig.suptitle("预测值 vs 真实 ΔSoC（7 个模型）", fontsize=18, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "06_scatter_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[plot_06]  完成 → plots/06_scatter_comparison.png")


if __name__ == "__main__":
    plot_scatter()
