#!/usr/bin/env python3
"""
Stride Ablation Study — Multi-trip evaluation with extended stride range.

Tests strides [75, 100, 120, 150, 180, 200, 250, 300] on 5 best trips.
Generates:
  10e_stride_ablation.png  — error metrics vs stride (per trip + average)
  10f_stride_tradeoff.png  — avg MAE vs avg prediction count (Pareto front)
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
Y_STD  = 1.6087
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
RAW_FEATURES  = {"Low", "Mid", "High", "cruising_ratio"}
CLIP_FEATURES = {"mileage_diff": (-5.0, 5.0)}

STRIDES   = [75, 100, 120, 150, 180, 200, 250, 300]
NUM_TRIPS = 5

STRIDE_COLORS = {
    75:  "#43A047", 100: "#66BB6A", 120: "#26A69A",
    150: "#E53935",
    180: "#FB8C00", 200: "#F4511E", 250: "#8E24AA", 300: "#1E88E5",
}

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
    """Sliding-window chain prediction for one trip / one stride."""
    n_rows = len(trip_df)
    n_windows = (n_rows - WINDOW_STEPS) // stride
    if n_windows < 1:
        return None

    all_windows = []
    for k in range(n_windows):
        start = k * stride
        end   = start + WINDOW_STEPS
        seg   = trip_df.iloc[start:end]
        X_raw = seg[FEATURE_COLUMNS].values.astype(np.float32)
        X_norm = normalize_features(X_raw, scaler_mean, scaler_std)
        all_windows.append(X_norm)

    X_batch = torch.from_numpy(np.stack(all_windows, axis=0)).to(device)
    with torch.no_grad():
        d_norm, _ = model(X_batch)
    d_raw = d_norm.cpu().numpy() * Y_STD + Y_MEAN

    init_soc = float(trip_df["refined_soc"].iloc[0])
    pred_t   = [0]
    pred_soc = [init_soc]
    true_at_pred = [init_soc]
    current_soc = init_soc

    for k in range(n_windows):
        d_partial = float(d_raw[k]) * (stride / WINDOW_STEPS)
        current_soc -= d_partial
        # For stride > WINDOW_STEPS, extrapolate to end of stride period.
        # For stride <= WINDOW_STEPS, predict at end of this window.
        if stride <= WINDOW_STEPS:
            predict_at = k * stride + WINDOW_STEPS
        else:
            predict_at = (k + 1) * stride
        true_soc = float(trip_df["refined_soc"].iloc[predict_at])
        pred_t.append(predict_at)
        pred_soc.append(current_soc)
        true_at_pred.append(true_soc)

    return (np.array(pred_t), np.array(pred_soc), np.array(true_at_pred))


def metrics(pred, true):
    err = pred - true
    return {
        "mae":     float(np.mean(np.abs(err))),
        "rmse":    float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(np.abs(err))),
        "n_pred":  len(pred),
    }


def evaluate_trip(model, trip_df, scaler_mean, scaler_std, device):
    """Score a trip by chain-prediction fit quality (stride=150 only)."""
    n_rows = len(trip_df)
    n_windows = (n_rows - WINDOW_STEPS) // 150
    if n_windows < 3:
        return None

    all_windows = []
    for k in range(n_windows):
        start = k * 150
        end   = start + WINDOW_STEPS
        seg   = trip_df.iloc[start:end]
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
        current_soc -= float(d_raw_batch[k])
        true_soc = float(trip_df["refined_soc"].iloc[(k + 1) * 150])
        errors.append(abs(current_soc - true_soc))

    errors = np.array(errors)
    soc_change = trip_df["refined_soc"].iloc[0] - trip_df["refined_soc"].iloc[-1]
    return {
        "trip_id":    int(trip_df["trip_id"].iloc[0]),
        "rows":       n_rows,
        "soc_change": soc_change,
        "mae":        float(np.mean(errors)),
        "rmse":       float(np.sqrt(np.mean(errors ** 2))),
        "rel_err":    float(np.mean(errors)) / max(abs(soc_change), 0.01) * 100,
    }


def select_top_trips(df, model, scaler_mean, scaler_std, device, n=NUM_TRIPS):
    """Return top N trips by chain-prediction fit quality."""
    trip_counts = df["trip_id"].value_counts()
    candidates = []

    for tid, cnt in trip_counts.items():
        if cnt >= 1000:
            tdf = df[df["trip_id"] == tid].reset_index(drop=True)
            soc_delta = tdf["refined_soc"].iloc[0] - tdf["refined_soc"].iloc[-1]
            if soc_delta >= 2:
                candidates.append((int(tid), tdf, cnt, soc_delta))

    print(f"  Scanning {len(candidates)} candidate trips (>=1000 rows, SoC Δ>=2%) ...")

    results = []
    for tid, tdf, cnt, soc_delta in candidates:
        r = evaluate_trip(model, tdf, scaler_mean, scaler_std, device)
        if r:
            results.append((r, tdf))

    if not results:
        raise RuntimeError("No qualified trips found.")

    results.sort(key=lambda x: x[0]["rel_err"])

    print(f"\n  Top {n} trips (by fit quality):")
    for i, (r, _) in enumerate(results[:n]):
        dur_h = r["rows"] * TIME_RES / 3600
        print(f"    #{i+1} Trip {r['trip_id']}: {r['rows']} rows ({dur_h:.1f}h), "
              f"SoC Δ={r['soc_change']:.2f}%, "
              f"MAE={r['mae']:.4f}, RelErr={r['rel_err']:.1f}%")

    return [(r["trip_id"], tdf) for r, tdf in results[:n]]


# ============================================================
# Figure 10e — Stride Ablation
# ============================================================

def plot_ablation(all_results, trip_ids):
    """
    Three subplots (MAE / RMSE / MaxErr) vs stride.
    Thin lines per trip, thick average line.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)

    metric_keys = ["mae", "rmse", "max_err"]
    metric_labels = ["MAE", "RMSE", "Max Error"]
    colors_trip = plt.cm.tab10(np.linspace(0, 1, len(trip_ids)))

    for ax_idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[ax_idx]
        all_vals = []

        for t_idx, tid in enumerate(trip_ids):
            vals = []
            for s in STRIDES:
                key = (tid, s)
                if key in all_results:
                    m = all_results[key]
                    vals.append(m[mkey])
                else:
                    vals.append(np.nan)
            all_vals.append(vals)
            ax.plot(STRIDES, vals, "o-", color=colors_trip[t_idx],
                    linewidth=1.2, markersize=5, alpha=0.7,
                    label=f"Trip #{tid}")

        # Average line
        avg_vals = np.nanmean(all_vals, axis=0)
        ax.plot(STRIDES, avg_vals, "s-", color="black",
                linewidth=2.8, markersize=9, label="Average", zorder=10)

        # Mark stride=150 (baseline)
        baseline_idx = STRIDES.index(150)
        ax.axvline(x=150, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.axhline(y=avg_vals[baseline_idx], color="gray",
                   linestyle=":", alpha=0.4, linewidth=1)

        # Annotate best point on average
        best_idx = np.nanargmin(avg_vals)
        best_s = STRIDES[best_idx]
        ax.annotate(f"best: stride={best_s}\n{mlabel}={avg_vals[best_idx]:.4f}",
                    xy=(best_s, avg_vals[best_idx]),
                    xytext=(best_s + 15, avg_vals[best_idx] * 1.15),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        ax.set_ylabel(f"{mlabel} (SoC %)", fontsize=12)
        ax.set_title(f"{mlabel} vs Stride", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Stride (steps, 1 step = 10s)", fontsize=13)
    axes[0].set_title(f"Figure 10e: Stride Ablation — Error Metrics  |  "
                      f"{len(trip_ids)} trips  |  S1500 Model",
                      fontsize=15, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "10e_stride_ablation.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [5/6] {path}")


# ============================================================
# Figure 10f — Tradeoff: MAE vs Prediction Count
# ============================================================

def plot_tradeoff(all_results, trip_ids):
    """
    X-axis: average number of prediction points per trip
    Y-axis: average MAE
    Each point = one stride, annotated.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    avg_mae  = []
    avg_npred = []

    for s in STRIDES:
        maes = []
        npreds = []
        for tid in trip_ids:
            key = (tid, s)
            if key in all_results:
                m = all_results[key]
                maes.append(m["mae"])
                npreds.append(m["n_pred"])
        avg_mae.append(np.mean(maes))
        avg_npred.append(np.mean(npreds))

    avg_mae  = np.array(avg_mae)
    avg_npred = np.array(avg_npred)

    # Draw curve
    ax.plot(avg_npred, avg_mae, "o-", color="#37474F", linewidth=2, markersize=10,
            markerfacecolor="white", markeredgewidth=2, zorder=2)

    # Annotate each point with stride
    offsets = {
        75:  (8, 0.02),  100: (8, -0.04), 120: (-35, -0.06),
        150: (-45, 0.03),
        180: (8, 0.04),   200: (8, -0.03), 250: (8, -0.06), 300: (-40, 0.05),
    }
    for s, x, y in zip(STRIDES, avg_npred, avg_mae):
        dx, dy = offsets.get(s, (10, 0.02))
        color = STRIDE_COLORS[s]
        ax.annotate(f"stride={s}",
                    xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=11, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.85))

    # Mark Pareto-optimal frontier
    pareto_x, pareto_y = compute_pareto(avg_npred, avg_mae)
    ax.plot(pareto_x, pareto_y, "--", color="green", linewidth=2.5, alpha=0.7,
            label="Pareto frontier")

    # Mark stride=150
    idx150 = STRIDES.index(150)
    ax.scatter([avg_npred[idx150]], [avg_mae[idx150]], s=200, c="red",
               marker="*", zorder=10, edgecolors="darkred", linewidths=1.5,
               label=f"stride=150 (no overlap)")

    ax.set_xlabel("Average Number of Prediction Points per Trip", fontsize=13)
    ax.set_ylabel("Average MAE (SoC %)", fontsize=13)
    ax.set_title(f"Figure 10f: Accuracy vs Update Frequency Tradeoff  |  "
                 f"{len(trip_ids)} trips  |  S1500 Model",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "10f_stride_tradeoff.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [6/6] {path}")


def compute_pareto(x, y):
    """Return Pareto frontier points (x is better when larger, y is better when smaller)."""
    idx = np.argsort(x)[::-1]  # sort by x descending
    xs, ys = x[idx], y[idx]
    pareto_x, pareto_y = [xs[0]], [ys[0]]
    best_y = ys[0]
    for i in range(1, len(xs)):
        if ys[i] < best_y:
            best_y = ys[i]
            pareto_x.append(xs[i])
            pareto_y.append(ys[i])
    return np.array(pareto_x), np.array(pareto_y)


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 65)
    print("  Stride Ablation — Multi-Trip Chain Prediction")
    print("  Strides:", STRIDES)
    print("  Trips:   top", NUM_TRIPS)
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Loading data ...")
    df = pd.read_csv(str(_PROJECT_ROOT / "data" / "processed" / "feature_data.csv"))
    print(f"  {df.shape[0]} rows, {df['trip_id'].nunique()} trips")

    # 2. Model & scaler
    print("\n[2/5] Loading model ...")
    model = load_model(device)
    scaler_mean, scaler_std = load_scaler()

    # 3. Select trips
    print("\n[3/5] Selecting top trips ...")
    selected = select_top_trips(df, model, scaler_mean, scaler_std, device, NUM_TRIPS)
    trip_ids = [tid for tid, _ in selected]

    # 4. Chain predictions — all trips × all strides
    print(f"\n[4/5] Running chain predictions "
          f"({len(selected)} trips × {len(STRIDES)} strides) ...")

    # Per-trip per-stride result dict
    all_results = {}

    # Summary table
    print(f"\n  {'Trip':>6s} {'Rows':>6s} {'SoCΔ':>7s}  " +
          "".join([f"{'s='+str(s):>13s}" for s in STRIDES]))
    print(f"  {'':->6s} {'':->6s} {'':->7s}  " +
          "".join([f"{'':->13s}" for _ in STRIDES]))

    for tid, tdf in selected:
        row_maes = []
        for s in STRIDES:
            res = chain_predict(model, tdf, s, scaler_mean, scaler_std, device)
            if res is not None:
                pred_t, pred_s, true_s = res
                m = metrics(pred_s, true_s)
                all_results[(tid, s)] = m
                row_maes.append(f"{m['mae']:.4f}")
            else:
                row_maes.append("N/A")
        soc_delta = tdf["refined_soc"].iloc[0] - tdf["refined_soc"].iloc[-1]
        print(f"  {tid:6d} {len(tdf):6d} {soc_delta:+6.2f}%  " +
              "".join([f"{v:>13s}" for v in row_maes]))

    # 5. Figures
    print("\n[5/5] Generating figures ...")
    plot_ablation(all_results, trip_ids)
    plot_tradeoff(all_results, trip_ids)

    # Print best stride recommendation
    print("\n" + "=" * 65)
    print("  Recommendation")
    print("=" * 65)
    for s in STRIDES:
        maes = [all_results[(tid, s)]["mae"] for tid in trip_ids if (tid, s) in all_results]
        npreds = [all_results[(tid, s)]["n_pred"] for tid in trip_ids if (tid, s) in all_results]
        if maes:
            print(f"  stride={s:3d}:  avg MAE={np.mean(maes):.4f}  "
                  f"avg points={np.mean(npreds):.1f}  "
                  f"range MAE=[{np.min(maes):.4f}, {np.max(maes):.4f}]")

    best_s = STRIDES[np.argmin([np.mean([all_results[(tid, s)]["mae"]
        for tid in trip_ids if (tid, s) in all_results]) for s in STRIDES])]
    print(f"\n  >>> Lowest MAE: stride={best_s}")

    print("\nDone. Output:")
    print(f"  plots/10e_stride_ablation.png")
    print(f"  plots/10f_stride_tradeoff.png")


if __name__ == "__main__":
    main()
