#!/usr/bin/env python3
"""
Plot 07: 残差分布 — 7 个模型，逐窗口推理。
布局: 4 行 × 2 列，零中心对称 x 轴。
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

# 中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from plot_utils import OUTPUT_DIR, WINDOWS, C
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


def run_residuals(model, device, data_dir, ymean, ystd):
    test_path = Path(data_dir) / "test_data.npz"
    data = np.load(test_path)
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
    residuals = y_pred_raw - y_true_raw
    return residuals, y_true_raw, y_pred_raw


def plot_residuals():
    device = "cpu"
    print(f"[plot_07] 设备: {device}")

    # 画布高度进一步加大，彻底不重叠
    fig, axes = plt.subplots(4, 2, figsize=(28, 40))
    axes = axes.flatten()

    all_residuals = []
    all_stats = []
    for w in WINDOWS:
        ckpt_path = _PROJECT_ROOT / "models" / f"best_model_{w}s.pt"
        data_dir  = _PROJECT_ROOT / "data" / "train" / f"split_{w}s"
        ymean, ystd = Y_STATS[w]

        model = load_model(ckpt_path, device)
        residuals, y_true, y_pred = run_residuals(model, device, data_dir, ymean, ystd)
        all_residuals.append(residuals)

        res_mean = float(np.mean(residuals))
        res_std  = float(np.std(residuals))
        all_stats.append((w, res_mean, res_std))

        del model
        torch.cuda.empty_cache()

    global_max_abs = max(np.abs(r).max() for r in all_residuals)
    xlim = (-global_max_abs * 1.1, global_max_abs * 1.1)

    for idx, (w, residuals, stats) in enumerate(zip(WINDOWS, all_residuals, all_stats)):
        ax = axes[idx]
        _, res_mean, res_std = stats

        ax.hist(
            residuals, bins=30, density=True, alpha=0.5,
            color=C["mae"], edgecolor="white", linewidth=1
        )

        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(residuals)
            xs = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(xs, kde(xs), color=C["r2"], linewidth=3)
        except Exception:
            pass

        ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.8)
        ax.set_xlim(*xlim)

        # ===================== 字号全部放大 =====================
        ax.set_xlabel("残差", fontsize=35, labelpad=12)
        ax.set_ylabel("概率密度", fontsize=35, labelpad=12)
        
        # 保留：窗口 + μ + σ，字号超大
        ax.set_title(
            f"窗口 {w}s    μ = {res_mean:.3f}    σ = {res_std:.3f}",
            fontsize=35, fontweight="bold", pad=20
        )
        
        ax.tick_params(labelsize=30)

    axes[-1].set_visible(False)

    # 总标题完全不重叠
    fig.suptitle("残差分布（7 个模型）", fontsize=40, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "07_residual_dist.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[plot_07]  完成 → plots/07_residual_dist.png")


if __name__ == "__main__":
    plot_residuals()