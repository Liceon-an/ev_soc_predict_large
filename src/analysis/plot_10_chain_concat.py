#!/usr/bin/env python3
"""
多段行程拼接链式预测 — 可视化

选取 7 条 trip，SoC 偏移拼接成连续曲线 (100%→36%, 12.9h)，
S1500 模型滑动窗口链式预测 ΔSoC 并累加重建 SoC 轨迹。

输出 (plots/):
  10a_full_timeline.png      — 12.9h 全时间线预测 vs 真实
  10b_zoom_3h.png            — 前 3h 放大
  10c_error_timeline.png     — 每步预测误差时序
  10d_scatter.png            — 预测值 vs 真实值散点图
  10e_metrics_by_trip.png    — 各段行程误差指标柱状图
  10f_error_accumulation.png — 累积误差增长曲线

用法:
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu
  python3 src/analysis/plot_10_chain_concat.py
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

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
WINDOW = 150          # S1500 模型时间步
STRIDE = 150          # 全窗口无重叠
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 17 维特征 (与 build_dataset.py 一致)
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

# 7 段行程 (来自 plan.md)
TRIP_IDS = [1166, 1643, 344, 744, 1125, 247, 1484]

# S1500 y_main 统计量
Y_MEAN = 2.1699
Y_STD = 1.6087

# 颜色
C_TRUE = "#2196F3"
C_PRED = "#F44336"
C_ERROR = "#FF5722"
C_ACCUM = "#9C27B0"
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
    """加载 feature_data.csv，返回全量 DataFrame"""
    print(f"[data] 加载 {_DATA_PATH}")
    df = pd.read_csv(_DATA_PATH)
    print(f"[data]   总行数: {len(df)}, trip_id 数: {df['trip_id'].nunique()}")
    return df


def load_scaler():
    """加载 S1500 scaler 参数"""
    scaler = np.load(_SCALER_PATH, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    print(f"[scaler] 已加载: {_SCALER_PATH}")
    return mean, std


def load_model():
    """加载 S1500 最佳模型"""
    ckpt = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = LSTMTransformer(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"[model] 已加载 S1500 (best epoch={ckpt['epoch']}), "
          f"参数量: {model.count_params():,}, device: {DEVICE}")
    return model


# ═══════════════════════════════════════════════════════════
# 特征提取与标准化
# ═══════════════════════════════════════════════════════════

def extract_trip_data(df, trip_id):
    """提取单条 trip 的特征数组和 refined_soc"""
    mask = df["trip_id"] == trip_id
    trip_df = df[mask].reset_index(drop=True)
    n = len(trip_df)

    X_raw = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        X_raw[:, i] = trip_df[col].values.astype(np.float32)

    refined_soc = trip_df["refined_soc"].values.astype(np.float32)
    time_col = trip_df.get("DATE", None)
    return X_raw, refined_soc, time_col, n


def apply_zscore(X_raw, mean, std):
    """对非 RAW_FEATURES 做 Z-Score，RAW_FEATURES 保持原值"""
    X = X_raw.copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        if col not in RAW_FEATURES:
            X[:, i] = (X[:, i] - mean[i]) / max(std[i], 1e-8)
        # RAW_FEATURES 保持原值
    return X


# ═══════════════════════════════════════════════════════════
# 链式预测核心
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def chain_predict_one_trip(model, X, refined_soc):
    """
    对单条 trip 执行链式预测。

    从 t=0 开始, 每 STRIDE 步取 X[t:t+WINDOW] 预测 ΔSoC,
    累加到 SoC_pred 中。初始 SoC 锚定为 refined_soc[0].

    Returns:
        pred_points: list of (idx, SoC_pred)  — 每个预测窗口终点
        SoC_true:    refined_soc 原始值
    """
    n = len(X)
    pred_points = []  # (absolute_index, SoC_pred_value)
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
    """计算每条 trip 的 SoC 偏移量，使 trip 间 SoC 连续"""
    offsets = []
    cum_offset = 0.0
    for i, (_, soc, _, _) in enumerate(trips_data):
        if i == 0:
            offsets.append(0.0)
        else:
            prev_soc = trips_data[i - 1][1]
            cum_offset += prev_soc[-1] - soc[0]
            offsets.append(cum_offset)
    return offsets


# ═══════════════════════════════════════════════════════════
# 绘图函数
# ═══════════════════════════════════════════════════════════

def _setup_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def _build_global_timeline(trips_data, offsets, all_preds):
    """
    构建全局时间线数据结构。

    Returns:
        global_true:   全局真实 SoC (逐点)
        global_x:      全局时间索引 (逐点)
        pred_x:        预测点时间索引
        pred_soc:      预测点 SoC 值
        trip_boundaries: 各段索引边界 [(start, end), ...]
    """
    global_true, global_x, pred_x, pred_soc = [], [], [], []
    boundaries = []
    cursor = 0

    for i, (X, soc, _, n) in enumerate(trips_data):
        offset = offsets[i]
        boundaries.append((cursor, cursor + n))
        for j in range(n):
            global_true.append(soc[j] + offset)
            global_x.append(cursor + j)
        for idx, val in all_preds[i]:
            pred_x.append(cursor + idx)
            pred_soc.append(val + offset)
        cursor += n

    return (
        np.array(global_true), np.array(global_x),
        np.array(pred_x), np.array(pred_soc),
        boundaries,
    )


def plot_10a_full_timeline(global_true, global_x, pred_x, pred_soc, boundaries):
    """图 10a: 12.9h 全时间线预测 vs 真实, 横坐标小时, 预测点连线"""
    # 转换为小时 (每步 10s, 360 步 = 1h)
    gx_h = global_x / 360.0
    px_h = pred_x / 360.0
    bd_h = [(s / 360.0, e / 360.0) for s, e in boundaries]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Trip 背景色 + 顶部标签
    y_min = float(global_true.min())
    y_max = float(global_true.max())
    y_pad = (y_max - y_min) * 0.05
    for i, (s, e) in enumerate(bd_h):
        ax.axvspan(s, e, alpha=0.12, color=TRIP_COLORS[i], zorder=0)
        mid = (s + e) / 2
        ax.text(mid, y_max + y_pad * 0.6, f"Trip {i + 1}", ha="center", fontsize=9,
                color=TRIP_COLORS_EDGE[i], fontweight="bold")

    # 真实 SoC — 连续蓝线
    ax.plot(gx_h, global_true, color=C_TRUE, linewidth=1.5, alpha=0.9,
            label="True SoC")

    # 预测点 — 红点 + 连线
    ax.plot(px_h, pred_soc, "o-", color=C_PRED, linewidth=2, markersize=7,
            markerfacecolor="white", markeredgewidth=1.5, zorder=5,
            label="Chain Prediction")

    # 行程分隔线
    for _, e in bd_h[:-1]:
        ax.axvline(e, color="#37474F", linestyle="--", linewidth=1, alpha=0.4)

    # X 轴: 小时刻度
    total_h = float(gx_h[-1])
    tick_step = 2.0  # 每 2 小时一个刻度
    xticks = np.arange(0, total_h + tick_step, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.0f}h" for t in xticks], fontsize=10)

    ax.set_xlabel("Elapsed Time (hours)", fontsize=12)
    ax.set_ylabel("SoC (%)", fontsize=12)
    ax.set_title("Multi-Trip Chain Prediction (12.9h, SoC 100% → 36%)", fontsize=15,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-0.1, float(gx_h[-1]) + 0.1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10a_full_timeline.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10a] 10a_full_timeline.png")


def plot_10b_zoom_3h(global_true, global_x, pred_x, pred_soc, boundaries):
    """图 10b: 前 3h 放大"""
    fig, ax = plt.subplots(figsize=(18, 6))

    zoom_end = 1080  # 前 3h ≈ 1080 步 (×10s)

    for i, (s, e) in enumerate(boundaries):
        if s >= zoom_end:
            break
        ax.axvspan(max(s, 0), min(e, zoom_end), alpha=0.15,
                   color=TRIP_COLORS[i], zorder=0)
        mid = (max(s, 0) + min(e, zoom_end)) / 2
        ax.text(mid, global_true[0] + 0.5, f"Trip {i + 1}", ha="center",
                fontsize=9, color=TRIP_COLORS_EDGE[i], fontweight="bold")

    mask_true = global_x <= zoom_end
    ax.plot(global_x[mask_true], global_true[mask_true], color=C_TRUE,
            linewidth=1.5, alpha=0.9, label="True SoC")

    mask_pred = pred_x <= zoom_end
    ax.scatter(pred_x[mask_pred], pred_soc[mask_pred], color=C_PRED, s=40,
               zorder=5, edgecolors="white", linewidth=0.5,
               label=f"Chain Prediction (n={mask_pred.sum()})")

    for _, e in boundaries[:-1]:
        if e <= zoom_end:
            ax.axvline(e, color="#37474F", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlabel("Timeline (sample index)")
    ax.set_ylabel("SoC (%)")
    ax.set_title("Chain Prediction — First 3 Hours (Zoom)", fontsize=15,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10b_zoom_3h.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10b] 10b_zoom_3h.png")


def plot_10c_error_timeline(pred_x, pred_soc, global_true, boundaries):
    """图 10c: 每步预测误差时序 (柱状图)"""
    errors = []
    for px, ps in zip(pred_x, pred_soc):
        idx = int(px)
        true_val = global_true[idx] if idx < len(global_true) else global_true[-1]
        errors.append(ps - true_val)

    fig, ax = plt.subplots(figsize=(18, 5))
    colors_bar = [C_ERROR if abs(e) < 1.0 else "#D32F2F" for e in errors]
    bars = ax.bar(pred_x, errors, width=30, color=colors_bar, edgecolor="white",
                  linewidth=0.3, alpha=0.85)

    # Trip 分界
    for _, e in boundaries[:-1]:
        ax.axvline(e, color="#37474F", linestyle="--", linewidth=0.8, alpha=0.3)

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(np.mean(errors), color="#E91E63", linestyle="--", linewidth=1.5,
               label=f"Mean Error = {np.mean(errors):.3f}")

    ax.set_xlabel("Timeline (sample index)")
    ax.set_ylabel("Prediction Error (ΔSoC)")
    ax.set_title("Chain Prediction Error Timeline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10c_error_timeline.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10c] 10c_error_timeline.png")


def plot_10d_scatter(pred_x, pred_soc, global_true, boundaries):
    """图 10d: 预测值 vs 真实值散点图"""
    pred_vals, true_vals = [], []
    for px, ps in zip(pred_x, pred_soc):
        idx = int(px)
        tv = global_true[idx] if idx < len(global_true) else global_true[-1]
        pred_vals.append(ps)
        true_vals.append(tv)

    pred_vals = np.array(pred_vals)
    true_vals = np.array(true_vals)
    errors = pred_vals - true_vals

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)

    fig, ax = plt.subplots(figsize=(9, 8))

    vmin = min(true_vals.min(), pred_vals.min())
    vmax = max(true_vals.max(), pred_vals.max())
    margin = (vmax - vmin) * 0.08 + 1.0
    lo = vmin - margin
    hi = vmax + margin

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="Perfect")

    sc = ax.scatter(true_vals, pred_vals, c=errors, cmap="coolwarm", s=60,
                    edgecolors="white", linewidth=0.5, zorder=5,
                    vmin=-1.5, vmax=1.5)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Error (pred − true)", fontsize=10)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("True SoC (%)")
    ax.set_ylabel("Predicted SoC (%)")
    ax.set_title(f"Chain Prediction Scatter\nR²={r2:.4f}  MAE={mae:.4f}  n={len(pred_vals)}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    ax.text(0.97, 0.05,
            f"Global MAE = {mae:.2f}%\nGlobal R² = {r2:.4f}",
            transform=ax.transAxes, fontsize=10, ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10d_scatter.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10d] 10d_scatter.png")


def plot_10e_metrics_by_trip(all_preds, trips_data, offsets, boundaries):
    """图 10e: 各段行程误差指标柱状图"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    trip_mae, trip_rmse, trip_maxerr, trip_n = [], [], [], []
    for i in range(len(trips_data)):
        preds_i = all_preds[i]
        _, soc, _, n = trips_data[i]
        offset = offsets[i]

        pred_vals, true_vals = [], []
        for idx, val in preds_i:
            tv = soc[idx] + offset if idx < len(soc) else soc[-1] + offset
            pred_vals.append(val + offset)
            true_vals.append(tv)

        if pred_vals:
            pred_vals = np.array(pred_vals)
            true_vals = np.array(true_vals)
            trip_mae.append(mean_absolute_error(true_vals, pred_vals))
            trip_rmse.append(np.sqrt(mean_squared_error(true_vals, pred_vals)))
            trip_maxerr.append(float(np.max(np.abs(pred_vals - true_vals))))
            trip_n.append(len(pred_vals))
        else:
            trip_mae.append(0)
            trip_rmse.append(0)
            trip_maxerr.append(0)
            trip_n.append(0)

    trip_labels = [f"T{i + 1}\n({n}pts)" for i, n in enumerate(trip_n)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE
    x = np.arange(len(trip_labels))
    wd = 0.6
    axes[0].bar(x, trip_mae, wd, color=TRIP_COLORS_EDGE, edgecolor="white")
    for i, v in enumerate(trip_mae):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trip_labels, fontsize=9)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE by Trip", fontsize=13, fontweight="bold")

    # RMSE
    axes[1].bar(x, trip_rmse, wd, color=TRIP_COLORS_EDGE, edgecolor="white")
    for i, v in enumerate(trip_rmse):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trip_labels, fontsize=9)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE by Trip", fontsize=13, fontweight="bold")

    # Max Error
    axes[2].bar(x, trip_maxerr, wd, color=TRIP_COLORS_EDGE, edgecolor="white")
    for i, v in enumerate(trip_maxerr):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(trip_labels, fontsize=9)
    axes[2].set_ylabel("Max |Error|")
    axes[2].set_title("Max Absolute Error by Trip", fontsize=13, fontweight="bold")

    fig.suptitle("Chain Prediction Error Metrics by Trip Segment", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10e_metrics_by_trip.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10e] 10e_metrics_by_trip.png")


def plot_10f_error_accumulation(pred_x, pred_soc, global_true):
    """图 10f: 累积误差增长曲线"""
    accum_errors = []  # 累积绝对误差
    cum_sum = 0.0
    for px, ps in zip(pred_x, pred_soc):
        idx = int(px)
        tv = global_true[idx] if idx < len(global_true) else global_true[-1]
        cum_sum += abs(ps - tv)
        accum_errors.append(cum_sum)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(pred_x, 0, accum_errors, alpha=0.3, color=C_ACCUM)
    ax.plot(pred_x, accum_errors, "o-", color=C_ACCUM, linewidth=2, ms=6,
            markerfacecolor="white")

    # 线性拟合看趋势
    z = np.polyfit(pred_x, accum_errors, 1)
    p = np.poly1d(z)
    ax.plot(pred_x, p(pred_x), "--", color="#E91E63", linewidth=1.5,
            label=f"Linear trend (slope={z[0]:.5f})")

    for i, (px, ae) in enumerate(zip(pred_x, accum_errors)):
        ax.annotate(f"{ae:.2f}", (px, ae), (0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color=C_ACCUM)

    ax.set_xlabel("Timeline (sample index)")
    ax.set_ylabel("Cumulative Absolute Error")
    ax.set_title("Error Accumulation Over Chain Prediction", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")

    ax.text(0.98, 0.05,
            f"Final cumulative: {accum_errors[-1]:.2f}\n"
            f"Avg error/step: {accum_errors[-1] / len(pred_x):.3f}",
            transform=ax.transAxes, fontsize=10, ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10f_error_accumulation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  [10f] 10f_error_accumulation.png")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print(f"[device] {DEVICE}")
    _setup_style()

    # 1. 加载数据
    df = load_data()
    mean, std = load_scaler()
    model = load_model()

    # 2. 提取各 trip 数据
    print(f"\n[trips] 提取 {len(TRIP_IDS)} 条行程...")
    trips_data = []  # [(X_raw, refined_soc, time_col, n), ...]
    for tid in TRIP_IDS:
        X_raw, soc, time_col, n = extract_trip_data(df, tid)
        print(f"  Trip {tid}: {n} rows, SoC [{soc[0]:.1f}% → {soc[-1]:.1f}%], Δ={soc[0] - soc[-1]:.1f}%")
        trips_data.append((X_raw, soc, time_col, n))

    # 3. 计算 SoC 偏移
    offsets = compute_trip_offsets(trips_data)
    print(f"\n[offsets] SoC 拼接偏移量:")
    for i, (tid, off) in enumerate(zip(TRIP_IDS, offsets)):
        _, soc, _, _ = trips_data[i]
        print(f"  Trip {tid}: offset={off:+.2f}, SoC [{soc[0] + off:.1f}% → {soc[-1] + off:.1f}%]")

    # 4. 链式预测
    print(f"\n[predict] 链式预测 (stride={STRIDE}, window={WINDOW})...")
    all_preds = []  # [[(idx, SoC_pred), ...], ...]
    total_pred_points = 0
    for i, (X_raw, soc, _, _) in enumerate(trips_data):
        X_z = apply_zscore(X_raw, mean, std)
        preds, _ = chain_predict_one_trip(model, X_z, soc)
        all_preds.append(preds)
        total_pred_points += len(preds)
        mae_i = np.mean([abs(p[1] - soc[p[0]]) for p in preds]) if preds else 0
        print(f"  Trip {TRIP_IDS[i]}: {len(preds)} pred points, MAE={mae_i:.4f}")

    print(f"  总预测点: {total_pred_points}")

    # 5. 构建全局时间线
    global_true, global_x, pred_x, pred_soc, boundaries = _build_global_timeline(
        trips_data, offsets, all_preds)

    total_delta = float(global_true[0] - global_true[-1])
    global_mae = float(np.mean(np.abs(
        np.interp(pred_x, global_x, global_true) - pred_soc
    )))
    print(f"\n[global] 总时长: {len(global_true) * 10 / 3600:.1f}h, "
          f"ΔSoC={total_delta:.1f}%, MAE={global_mae:.4f}, "
          f"相对误差={global_mae / total_delta * 100:.2f}%")

    # 6. 生成 6 张图
    print("\n[plot] 生成图片...")
    plot_10a_full_timeline(global_true, global_x, pred_x, pred_soc, boundaries)
    plot_10b_zoom_3h(global_true, global_x, pred_x, pred_soc, boundaries)
    plot_10c_error_timeline(pred_x, pred_soc, global_true, boundaries)
    plot_10d_scatter(pred_x, pred_soc, global_true, boundaries)
    plot_10e_metrics_by_trip(all_preds, trips_data, offsets, boundaries)
    plot_10f_error_accumulation(pred_x, pred_soc, global_true)

    print(f"\n{'=' * 50}")
    print(f"  全部完成! 输出: {_OUTPUT_DIR}/10[a-f]_*.png")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
