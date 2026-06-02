#!/usr/bin/env python3
"""
多段行程拼接链式预测 — 可视化（仅输出 10a 全时间线图）
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

# ─── 路径配置 ─────────────────────────────────────────────
_PROJECT_ROOT = Path("/root/code/ev_soc_predict")
_DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "feature_data.csv"
_MODEL_PATH = _PROJECT_ROOT / "models" / "best_model_1500s.pt"
_SCALER_PATH = _PROJECT_ROOT / "data" / "train" / "split_1500s" / "scaler.npz"
_OUTPUT_DIR = _PROJECT_ROOT / "plots"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_PROJECT_ROOT))

from configs.model_config import model_config
from src.models.lstm_transformer import LSTMTransformer

# ─── 常量 ─────────────────────────────────────────────────
WINDOW = 150
STRIDE = 150
DEVICE = "cpu"

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

TRIP_IDS = [1166, 1643, 344, 744, 1125, 247, 1484]

Y_MEAN = 2.1699
Y_STD = 1.6087

C_TRUE = "#2196F3"
C_PRED = "#F44336"
TRIP_COLORS = [
    "#E3F2FD", "#BBDEFB", "#B2EBF2", "#C8E6C9",
    "#FFF9C4", "#FFE0B2", "#F8BBD0",
]
TRIP_COLORS_EDGE = [
    "#2196F3", "#1976D2", "#00BCD4", "#4CAF50",
    "#FFC107", "#FF9800", "#E91E63",
]

# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════
def load_data():
    print(f"[data] 加载 {_DATA_PATH}")
    df = pd.read_csv(_DATA_PATH)
    print(f"[data]   总行数: {len(df)}, trip_id 数: {df['trip_id'].nunique()}")
    return df

def load_scaler():
    scaler = np.load(_SCALER_PATH, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    print(f"[scaler] 已加载")
    return mean, std

def load_model():
    ckpt = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = LSTMTransformer(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        model.to(DEVICE)
    except:
        model.to("cpu")
    model.eval()
    print(f"[model] 已加载 S1500 模型")
    return model

# ═══════════════════════════════════════════════════════════
# 特征提取
# ═══════════════════════════════════════════════════════════
def extract_trip_data(df, trip_id):
    mask = df["trip_id"] == trip_id
    trip_df = df[mask].reset_index(drop=True)
    n = len(trip_df)
    X_raw = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        X_raw[:, i] = trip_df[col].values.astype(np.float32)
    refined_soc = trip_df["refined_soc"].values.astype(np.float32)
    return X_raw, refined_soc, n

def apply_zscore(X_raw, mean, std):
    X = X_raw.copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        if col not in RAW_FEATURES:
            X[:, i] = (X[:, i] - mean[i]) / max(std[i], 1e-8)
    return X

# ═══════════════════════════════════════════════════════════
# 链式预测
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def chain_predict_one_trip(model, X, refined_soc):
    n = len(X)
    pred_points = []
    current_soc = float(refined_soc[0])
    for t in range(0, n - WINDOW, STRIDE):
        window_x = torch.from_numpy(X[t : t + WINDOW]).float().unsqueeze(0).to(DEVICE)
        delta_norm, _ = model(window_x)
        delta_pred = float(delta_norm.cpu().item()) * Y_STD + Y_MEAN
        current_soc -= delta_pred
        pred_points.append((t + WINDOW, current_soc))
    return pred_points, refined_soc.copy()

# ═══════════════════════════════════════════════════════════
# SoC 拼接
# ═══════════════════════════════════════════════════════════
def compute_trip_offsets(trips_data):
    offsets = []
    cum_offset = 0.0
    for i, (_, soc, _) in enumerate(trips_data):
        if i == 0:
            offsets.append(0.0)
        else:
            prev_soc = trips_data[i-1][1]
            cum_offset += prev_soc[-1] - soc[0]
            offsets.append(cum_offset)
    return offsets

def _build_global_timeline(trips_data, offsets, all_preds):
    global_true, global_x, pred_x, pred_soc = [], [], [], []
    boundaries = []
    cursor = 0
    for i, (X, soc, n) in enumerate(trips_data):
        offset = offsets[i]
        boundaries.append((cursor, cursor + n))
        for j in range(n):
            global_true.append(soc[j] + offset)
            global_x.append(cursor + j)
        for idx, val in all_preds[i]:
            pred_x.append(cursor + idx)
            pred_soc.append(val + offset)
        cursor += n
    return np.array(global_true), np.array(global_x), np.array(pred_x), np.array(pred_soc), boundaries

# ═══════════════════════════════════════════════════════════
# 仅保留：10a 全时间线绘图
# ═══════════════════════════════════════════════════════════
def _setup_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    plt.rcParams.update({
        "figure.dpi": 150, "font.size":13, "axes.titlesize":17,
        "axes.labelsize":15, "legend.fontsize":12
    })

def plot_10a_full_timeline(global_true, global_x, pred_x, pred_soc, boundaries):
    import matplotlib.pyplot as plt
    gx_h = global_x / 360.0
    px_h = pred_x / 360.0
    bd_h = [(s/360.0, e/360.0) for s,e in boundaries]

    fig, ax = plt.subplots(figsize=(16,6))
    y_min = float(global_true.min())
    y_max = float(global_true.max())
    y_pad = (y_max - y_min)*0.05

    for i, (s,e) in enumerate(bd_h):
        ax.axvspan(s,e, alpha=0.12, color=TRIP_COLORS[i], zorder=0)
        mid = (s+e)/2
        ax.text(mid, y_max + y_pad*0.6 -5, f"行程 {i+1}", ha="center", fontsize=18,
                color=TRIP_COLORS_EDGE[i], fontweight="bold")

    ax.plot(gx_h, global_true, color=C_TRUE, linewidth=2.0, alpha=0.9, label="真实 SOC")
    ax.plot(px_h, pred_soc, "o-", color=C_PRED, linewidth=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=1.5, zorder=5, label="链式预测")

    for _, e in bd_h[:-1]:
        ax.axvline(e, color="#37474F", linestyle="--", linewidth=1, alpha=0.4)

    total_h = float(gx_h[-1])
    tick_step = 2.0
    xticks = np.arange(0, total_h + tick_step, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.0f}h" for t in xticks], fontsize=18)
    ax.set_xlabel("经过时间（小时）", fontsize=20)
    ax.set_ylabel("SOC（%）", fontsize=20)
    ax.set_title("多段行程链式预测（SOC 100% → 36%）", fontsize=25, fontweight="bold")
    ax.legend(loc="lower left", fontsize=18)
    ax.set_xlim(-0.1, float(gx_h[-1])+0.1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.tick_params(labelsize=18)

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10a_full_timeline.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10a] 已保存：10a_full_timeline.png")

# ═══════════════════════════════════════════════════════════
# 主流程（仅画 10a）
# ═══════════════════════════════════════════════════════════
def main():
    print(f"[device] {DEVICE}")
    _setup_style()
    df = load_data()
    mean, std = load_scaler()
    model = load_model()

    print(f"\n[trips] 提取 {len(TRIP_IDS)} 条行程...")
    trips_data = []
    for tid in TRIP_IDS:
        X_raw, soc, n = extract_trip_data(df, tid)
        trips_data.append((X_raw, soc, n))

    offsets = compute_trip_offsets(trips_data)
    print(f"\n[predict] 链式预测中...")

    all_preds = []
    for i, (X_raw, soc, _) in enumerate(trips_data):
        X_z = apply_zscore(X_raw, mean, std)
        preds, _ = chain_predict_one_trip(model, X_z, soc)
        all_preds.append(preds)

    global_true, global_x, pred_x, pred_soc, boundaries = _build_global_timeline(trips_data, offsets, all_preds)
    print("\n[plot] 生成 10a 全时间线图...")
    plot_10a_full_timeline(global_true, global_x, pred_x, pred_soc, boundaries)

    print(f"\n✅ 完成！仅输出 10a 图")

if __name__ == "__main__":
    main()