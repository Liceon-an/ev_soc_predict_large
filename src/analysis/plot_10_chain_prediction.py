#!/usr/bin/env python3
"""
Chain Prediction — 4 separate figures with GPU-accelerated trip search.

Fig1: Three subplots, one per stride, predicted vs true.
Fig2: All three strides overlaid (summary).
Fig3: Error over time per stride.
Fig4: Metrics bar chart.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path("/root/code/ev_soc_predict")
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from configs.model_config import model_config
from src.models.lstm_transformer import LSTMTransformer

# ============================================================
# Config
# ============================================================
WINDOW_STEPS = 150
Y_MEAN = 2.1699
Y_STD = 1.6087
TIME_RES = 10

FEATURE_COLUMNS = [
    "speed", "speed_diff", "mileage_diff",
    "speed_window20_mean", "speed_diff_window20_mean",
    "temperature_c_window20_mean", "relative_humidity_window20_mean",
    "visibility_km_window20_mean", "wind_speed_ms_window20_mean",
    "speed_window20_std",
    "Low", "Mid", "High", "cruising_ratio",
    "total_volt", "total_current", "power",
]
RAW_FEATURES = {"Low", "Mid", "High", "cruising_ratio"}
CLIP_FEATURES = {"mileage_diff": (-5.0, 5.0)}

STRIDES = [150, 75, 15]
STRIDE_LABELS = {
    150: "stride=150 (full window)",
    75:  "stride=75  (50% overlap)",
    15:  "stride=15  (fine grain)",
}
COLORS = {150: "#E53935", 75: "#43A047", 15: "#1E88E5"}

OUTPUT_DIR = _PROJECT_ROOT / "plots"

# ============================================================
# Helpers
# ============================================================

def load_model(device):
    ckpt = torch.load(str(_PROJECT_ROOT / "models" / "best_model.pt"),
                      map_location="cpu", weights_only=False)
    model = LSTMTransformer(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_scaler():
    data = np.load(str(_PROJECT_ROOT / "data" / "train" / "split_1500s" / "scaler.npz"),
                   allow_pickle=True)
    return data["mean"], data["std"]


def normalize_features(X_raw, mean, std):
    X = X_raw.copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        if col not in RAW_FEATURES:
            X[:, i] = (X[:, i] - mean[i]) / std[i]
    for col, (lo, hi) in CLIP_FEATURES.items():
        i = FEATURE_COLUMNS.index(col)
        X[:, i] = np.clip(X[:, i], lo, hi)
    return X


def chain_predict(model, trip_df, stride, scaler_mean, scaler_std, device):
    n_rows = len(trip_df)
    n_windows = (n_rows - WINDOW_STEPS) // stride

    # Batch all windows for GPU
    all_windows = []
    for k in range(n_windows):
        start = k * stride
        end = start + WINDOW_STEPS
        seg = trip_df.iloc[start:end]
        X_raw = seg[FEATURE_COLUMNS].values.astype(np.float32)
        X_norm = normalize_features(X_raw, scaler_mean, scaler_std)
        all_windows.append(X_norm)

    X_batch = torch.from_numpy(np.stack(all_windows, axis=0)).to(device)
    with torch.no_grad():
        d_norm, _ = model(X_batch)
    d_raw = d_norm.cpu().numpy() * Y_STD + Y_MEAN

    init_soc = float(trip_df["refined_soc"].iloc[0])
    pred_t = [0]
    pred_soc = [init_soc]
    true_at_pred = [init_soc]
    current_soc = init_soc

    for k in range(n_windows):
        d_partial = float(d_raw[k]) * (stride / WINDOW_STEPS)
        current_soc = current_soc - d_partial
        predict_at = k * stride + WINDOW_STEPS  # end of this window
        true_soc = float(trip_df["refined_soc"].iloc[predict_at])
        pred_t.append(predict_at)
        pred_soc.append(current_soc)
        true_at_pred.append(true_soc)

    return (np.array(pred_t), np.array(pred_soc), np.array(true_at_pred))


def metrics(pred, true):
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(np.abs(err))),
    }


def evaluate_trip(model, trip_df, scaler_mean, scaler_std, device):
    """Quick evaluation of chain prediction quality on this trip (stride=150 only)."""
    n_rows = len(trip_df)
    n_windows = (n_rows - WINDOW_STEPS) // 150
    if n_windows < 3:
        return None

    all_windows = []
    for k in range(n_windows):
        start = k * 150
        end = start + WINDOW_STEPS
        seg = trip_df.iloc[start:end]
        X_raw = seg[FEATURE_COLUMNS].values.astype(np.float32)
        X_norm = normalize_features(X_raw, scaler_mean, scaler_std)
        all_windows.append(X_norm)

    X_batch = torch.from_numpy(np.stack(all_windows, axis=0)).to(device)
    with torch.no_grad():
        d_norm, _ = model(X_batch)
    d_raw_batch = d_norm.cpu().numpy() * Y_STD + Y_MEAN

    init_soc = float(trip_df["refined_soc"].iloc[0])
    current_soc = init_soc
    errors = []

    for k in range(n_windows):
        current_soc = current_soc - float(d_raw_batch[k])
        true_soc = float(trip_df["refined_soc"].iloc[(k + 1) * 150])
        errors.append(abs(current_soc - true_soc))

    errors = np.array(errors)
    soc_change = trip_df["refined_soc"].iloc[0] - trip_df["refined_soc"].iloc[-1]
    return {
        "trip_id": int(trip_df["trip_id"].iloc[0]),
        "rows": n_rows,
        "soc_change": soc_change,
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "rel_err": float(np.mean(errors)) / max(abs(soc_change), 0.01) * 100,
    }


def select_best_trip(df, model, scaler_mean, scaler_std, device):
    """Find a long trip with clear SoC trend and good prediction fit."""
    trip_counts = df["trip_id"].value_counts()
    candidates = []

    for tid, cnt in trip_counts.items():
        if cnt >= 1000:
            tdf = df[df["trip_id"] == tid].reset_index(drop=True)
            soc_delta = tdf["refined_soc"].iloc[0] - tdf["refined_soc"].iloc[-1]
            if soc_delta >= 2:
                candidates.append((int(tid), tdf, cnt, soc_delta))

    print(f"  Scanning {len(candidates)} candidate trips...")

    results = []
    for i, (tid, tdf, cnt, soc_delta) in enumerate(candidates):
        r = evaluate_trip(model, tdf, scaler_mean, scaler_std, device)
        if r:
            results.append(r)

    if not results:
        # fallback: longest trip
        tid = int(trip_counts.index[0])
        tdf = df[df["trip_id"] == tid].reset_index(drop=True)
        return tid, tdf, 0

    # Sort by rel_err (lower = better fit)
    results.sort(key=lambda x: x["rel_err"])

    print(f"\n  Top candidates (by fit quality):")
    for i, r in enumerate(results[:3]):
        duration_h = r["rows"] * TIME_RES / 3600
        print(f"    #{i+1} Trip {r['trip_id']}: {r['rows']} rows ({duration_h:.1f}h), "
              f"SoC Δ={r['soc_change']:.2f}%, MAE={r['mae']:.4f}, RelErr={r['rel_err']:.1f}%")

    best = results[0]
    trip_id = best["trip_id"]
    trip_df = df[df["trip_id"] == trip_id].reset_index(drop=True)
    return trip_id, trip_df, best["rel_err"]


# ============================================================
# Figure 1: Three subplots, one per stride
# ============================================================

def plot_fig1(trip_df, all_results, trip_id):
    time_axis = np.arange(len(trip_df)) * TIME_RES
    true_soc = trip_df["refined_soc"].values

    fig, axes = plt.subplots(3, 1, figsize=(18, 18), sharex=True)
    fig.suptitle(f"Figure 1: Chain Prediction — Per-Stride Comparison  |  Trip #{trip_id}",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, stride in enumerate(STRIDES):
        ax = axes[idx]
        pred_t, pred_s, true_s = all_results[stride]
        m = metrics(pred_s, true_s)

        ax.plot(time_axis, true_soc, color="gray", linewidth=1.2, alpha=0.6,
                label="Ground truth (refined_soc)", zorder=1)
        ax.plot(pred_t * TIME_RES, pred_s, "o-", color=COLORS[stride],
                linewidth=2.2, markersize=6, label="Chain prediction", zorder=2)
        ax.scatter(pred_t * TIME_RES, true_s, c="black", s=25, alpha=0.5,
                   marker="x", label="True at pred. point", zorder=3)
        ax.fill_between(pred_t * TIME_RES, pred_s, true_s, alpha=0.12,
                        color=COLORS[stride])

        ax.set_ylabel("SoC (%)", fontsize=12)
        ax.set_title(f"{STRIDE_LABELS[stride]}  |  "
                     f"Predictions: {len(pred_t)}  |  "
                     f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MaxErr={m['max_err']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[-1].set_xlabel("Time (seconds)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / "10a_per_stride_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [1/4] {path}")


# ============================================================
# Figure 2: All three strides overlaid + zoom
# ============================================================

def plot_fig2(trip_df, all_results, trip_id):
    time_axis = np.arange(len(trip_df)) * TIME_RES
    true_soc = trip_df["refined_soc"].values

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

    # Left: full timeline
    ax_full = fig.add_subplot(gs[0, 0])
    ax_full.plot(time_axis, true_soc, color="gray", linewidth=1.5, alpha=0.5,
                 label="Ground truth", zorder=1)
    for stride in STRIDES:
        pred_t, pred_s, _ = all_results[stride]
        ax_full.plot(pred_t * TIME_RES, pred_s, "o-", color=COLORS[stride],
                     linewidth=2, markersize=5, label=STRIDE_LABELS[stride], zorder=2)
    ax_full.set_xlabel("Time (seconds)", fontsize=12)
    ax_full.set_ylabel("SoC (%)", fontsize=12)
    ax_full.set_title("Full Timeline — All Three Strides", fontsize=14, fontweight="bold")
    ax_full.legend(loc="upper right", fontsize=9)
    ax_full.grid(True, alpha=0.25)

    # Right: first 6000s zoom
    ax_zoom = fig.add_subplot(gs[0, 1])
    zoom_end = min(6000, len(trip_df) * TIME_RES)
    mask = time_axis <= zoom_end
    ax_zoom.plot(time_axis[mask], true_soc[mask], color="gray", linewidth=1.5,
                 alpha=0.5, label="Ground truth", zorder=1)
    for stride in STRIDES:
        pred_t, pred_s, _ = all_results[stride]
        m = pred_t * TIME_RES <= zoom_end
        ax_zoom.plot(pred_t[m] * TIME_RES, pred_s[m], "o-", color=COLORS[stride],
                     linewidth=2.2, markersize=6, label=STRIDE_LABELS[stride], zorder=2)
    ax_zoom.set_xlabel("Time (seconds)", fontsize=12)
    ax_zoom.set_ylabel("SoC (%)", fontsize=12)
    ax_zoom.set_title(f"First {int(zoom_end)}s — Zoom", fontsize=14, fontweight="bold")
    ax_zoom.legend(loc="upper right", fontsize=9)
    ax_zoom.grid(True, alpha=0.25)

    fig.suptitle(f"Figure 2: Chain Prediction Summary  |  Trip #{trip_id}  |  S1500 Model",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUTPUT_DIR / "10b_summary_overlay.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [2/4] {path}")


# ============================================================
# Figure 3: Error over time per stride
# ============================================================

def plot_fig3(trip_df, all_results, trip_id):
    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    fig.suptitle(f"Figure 3: Prediction Error Over Time  |  Trip #{trip_id}",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, stride in enumerate(STRIDES):
        ax = axes[idx]
        pred_t, pred_s, true_s = all_results[stride]
        error = pred_s - true_s
        m = metrics(pred_s, true_s)

        bar_width = max(30, stride * TIME_RES * 0.65)
        colors_bar = ["#E53935" if e > 0 else "#1E88E5" for e in error]

        ax.bar(pred_t * TIME_RES, error, width=bar_width, color=colors_bar,
               alpha=0.7, edgecolor="none")
        ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-")
        ax.axhspan(-m["mae"], m["mae"], alpha=0.08, color="orange")
        ax.axhline(y=m["mae"], color="orange", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(y=-m["mae"], color="orange", linewidth=0.8, linestyle="--", alpha=0.7,
                    label=f"±MAE = ±{m['mae']:.4f}")

        ax.set_ylabel("Error (SoC %)", fontsize=12)
        ax.set_title(f"{STRIDE_LABELS[stride]}  |  "
                     f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MaxErr={m['max_err']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    axes[-1].set_xlabel("Time (seconds)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / "10c_error_timeline.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [3/4] {path}")


# ============================================================
# Figure 4: Metrics bar chart
# ============================================================

def plot_fig4(all_results, trip_id):
    fig, ax = plt.subplots(figsize=(14, 8))

    metric_names = ["MAE", "RMSE", "MaxErr"]
    x = np.arange(len(metric_names))
    width = 0.22

    for i, stride in enumerate(STRIDES):
        _, pred_s, true_s = all_results[stride]
        m = metrics(pred_s, true_s)
        values = [m["mae"], m["rmse"], m["max_err"]]
        bars = ax.bar(x + i * width, values, width, label=STRIDE_LABELS[stride],
                      color=COLORS[stride], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontsize=14)
    ax.set_ylabel("Error (SoC percentage points)", fontsize=13)
    ax.set_title(f"Figure 4: Error Metrics by Stride  |  Trip #{trip_id}  |  S1500 Model",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "10d_metrics_bar.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [4/4] {path}")


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Chain Prediction — 4-Figure Visualization")
    print("=" * 60)

    # 1. Data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(str(_PROJECT_ROOT / "data" / "processed" / "feature_data.csv"))
    print(f"  {df.shape[0]} rows, {df['trip_id'].nunique()} trips")

    # 2. Model & scaler
    print("\n[2/4] Loading model & searching for best trip...")
    model = load_model(device)
    scaler_mean, scaler_std = load_scaler()

    # 3. Select trip
    trip_id, trip_df, rel_err = select_best_trip(df, model, scaler_mean, scaler_std, device)
    duration_h = len(trip_df) * TIME_RES / 3600
    soc_start = trip_df["refined_soc"].iloc[0]
    soc_end = trip_df["refined_soc"].iloc[-1]
    soc_change = soc_start - soc_end
    print(f"\n  Selected: Trip #{trip_id}: {len(trip_df)} rows ({duration_h:.1f}h)")
    print(f"  SoC: {soc_start:.2f}% -> {soc_end:.2f}% (Δ = {soc_change:.2f}%)")

    # 4. Chain predictions
    print("\n[3/4] Running chain predictions (GPU batched)...")
    all_results = {}
    for stride in STRIDES:
        n_win = (len(trip_df) - WINDOW_STEPS) // stride
        results = chain_predict(model, trip_df, stride, scaler_mean, scaler_std, device)
        all_results[stride] = results
        m = metrics(results[1], results[2])
        print(f"  stride={stride:3d}: {len(results[0]):4d} points  "
              f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MaxErr={m['max_err']:.4f}")

    # 5. Plots
    print("\n[4/4] Generating 4 figures...")
    plot_fig1(trip_df, all_results, trip_id)
    plot_fig2(trip_df, all_results, trip_id)
    plot_fig3(trip_df, all_results, trip_id)
    plot_fig4(all_results, trip_id)

    print("\nDone. Output:")
    for f in ["10a_per_stride_comparison.png", "10b_summary_overlay.png",
              "10c_error_timeline.png", "10d_metrics_bar.png"]:
        print(f"  plots/{f}")


if __name__ == "__main__":
    main()
