#!/usr/bin/env python3
"""
特征影响分析 — 核心计算模块

四阶段分析:
  Phase 1: 特征-ΔSoC 相关分析 (Spearman + Pearson)
  Phase 2: Permutation Importance (逐特征 shuffle，测 R² 下降)
  Phase 3: 特征组贡献 (Group Permutation)
  Phase 4: 综合结论 (由 plot 脚本完成)

用法:
  from feature_analysis import run_phase1, run_phase2, run_phase3
  result1 = run_phase1()  # 相关性
  result2 = run_phase2()  # Permutation Importance
  result3 = run_phase3()  # Group Permutation
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/root/code/ev_soc_predict")
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import model_config
from configs.train_config import train_config
from src.models.lstm_transformer import LSTMTransformer

# ──────────────────────────────────────────────────────
# 17 个特征的定义（与 build_dataset.py 严格一致）
# ──────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "speed", "speed_diff", "mileage_diff",
    "speed_window20_mean", "speed_diff_window20_mean",
    "temperature_c_window20_mean", "relative_humidity_window20_mean",
    "visibility_km_window20_mean", "wind_speed_ms_window20_mean",
    "speed_window20_std",
    "Low", "Mid", "High", "cruising_ratio",
    "total_volt", "total_current", "power",
]

FEATURE_GROUPS = {
    "Speed (瞬时)":   ["speed", "speed_diff", "speed_window20_std"],
    "Speed (窗口)":   ["speed_window20_mean", "speed_diff_window20_mean"],
    "Weather":        ["temperature_c_window20_mean", "relative_humidity_window20_mean",
                       "visibility_km_window20_mean", "wind_speed_ms_window20_mean"],
    "Driving Mode":   ["Low", "Mid", "High", "cruising_ratio"],
    "Electrical":     ["total_volt", "total_current", "power"],
    "Mileage":        ["mileage_diff"],
}

FEAT_TO_GROUP = {}
for gname, feats in FEATURE_GROUPS.items():
    for f in feats:
        FEAT_TO_GROUP[f] = gname

# 实验窗口
WINDOW = "1500"
DATA_DIR = PROJECT_ROOT / "data" / "train" / f"split_{WINDOW}s"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
SCALER_PATH = DATA_DIR / "scaler.npz"
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────
# 数据与模型加载
# ──────────────────────────────────────────────────────

def load_test_data():
    """加载测试集 X (已 Z-score) 和 y_main (原始 ΔSoC)"""
    test_data = np.load(DATA_DIR / "test_data.npz")
    X_test = test_data["X"].astype(np.float32)           # (N, T, F)
    y_test = test_data["y_main"].astype(np.float32)      # (N,) raw ΔSoC

    train_data = np.load(DATA_DIR / "train_data.npz")
    y_mean = float(train_data["y_main"].mean())
    y_std  = float(train_data["y_main"].std())

    print(f"[data] 测试集: {X_test.shape[0]} 样本, "
          f"T={X_test.shape[1]}, F={X_test.shape[2]}")
    print(f"[data] y_main(ΔSoC): mean={y_mean:.4f}, std={y_std:.4f}")
    return X_test, y_test, y_mean, y_std


def load_model(device="cpu"):
    """加载 S1500 最佳模型"""
    train_config.device = device
    model = LSTMTransformer(model_config)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[model] 已加载 S{WINDOW}s 最佳模型 (best epoch={ckpt['epoch']})")
    print(f"[model] 参数量: {model.count_params():,}")
    return model


@torch.no_grad()
def predict(model, X, y_mean, y_std, device="cpu"):
    """
    批量推理，返回原始量纲的预测值 (N,)
    X: (N, T, F) — 已 Z-Score 标准化
    """
    X_t = torch.from_numpy(X).float().to(device)
    pred_norm, _ = model(X_t)
    pred_raw = pred_norm.cpu().numpy() * y_std + y_mean
    return pred_raw


def batch_evaluate(model, X, y_true, y_mean, y_std, device="cpu"):
    """返回 R² 和 MAE"""
    pred = predict(model, X, y_mean, y_std, device)
    r2 = r2_score(y_true, pred)
    mae = float(np.mean(np.abs(pred - y_true)))
    return r2, mae, pred


def get_baseline(model, X_test, y_test, y_mean, y_std, device="cpu"):
    """获取测试集基线 R²"""
    r2, mae, pred = batch_evaluate(model, X_test, y_test, y_mean, y_std, device)
    print(f"[baseline] Test R² = {r2:.4f}, MAE = {mae:.4f}")
    return r2, mae


# ──────────────────────────────────────────────────────
# Phase 1: 特征相关性分析
# ──────────────────────────────────────────────────────

def _aggregate_features(X):
    """
    对每个样本的时间维度做聚合，返回 (N, F)。
    用均值聚合 + 最后一步值 + 最大值，组合成丰富表征。
    """
    N, T, F = X.shape
    pooled = np.zeros((N, F), dtype=np.float32)
    for f in range(F):
        col = FEATURE_COLUMNS[f]
        # RAW_FEATURES (比例类) 用最后一步
        if col in ("Low", "Mid", "High", "cruising_ratio"):
            pooled[:, f] = X[:, -1, f]
        else:
            pooled[:, f] = X[:, :, f].mean(axis=1)
    return pooled


def compute_feature_target_corr(X, y):
    """
    计算每个特征与 ΔSoC 的 Spearman + Pearson 相关系数。
    返回 DataFrame: feature | pearson_r | spearman_r | p_value | group
    """
    pooled = _aggregate_features(X)
    records = []
    for i, col in enumerate(FEATURE_COLUMNS):
        sp = spearmanr(pooled[:, i], y)
        pr = pearsonr(pooled[:, i], y)
        records.append({
            "feature": col,
            "group": FEAT_TO_GROUP.get(col, "Other"),
            "pearson_r": pr.statistic,
            "pearson_p": pr.pvalue,
            "spearman_r": sp.statistic,
            "spearman_p": sp.pvalue,
        })
    df = pd.DataFrame(records)
    return df, pooled


def compute_feature_corr_matrix(X):
    """计算特征间 Spearman 相关矩阵"""
    pooled = _aggregate_features(X)
    corr_matrix, p_matrix = spearmanr(pooled, axis=0, nan_policy='omit')
    # spearmanr returns (F, F) matrix
    if corr_matrix.ndim == 0:
        corr_matrix = np.array([[1.0]])
    # Replace NaN (from constant features) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix, FEATURE_COLUMNS


# ──────────────────────────────────────────────────────
# Phase 2: Permutation Importance
# ──────────────────────────────────────────────────────

def permutation_importance(model, X, y, y_mean, y_std,
                           n_repeats=5, device="cpu", seed=42):
    """
    逐特征 shuffle，计算 R² 下降。

    Args:
        n_repeats: 每特征重复 shuffle 次数（取均值和标准差）
    Returns:
        DataFrame: feature | importance_mean | importance_std | group
    """
    rng = np.random.default_rng(seed)
    baseline_r2, _ = get_baseline(model, X, y, y_mean, y_std, device)
    r2_drop = np.zeros((len(FEATURE_COLUMNS), n_repeats), dtype=np.float32)

    for i, col in enumerate(FEATURE_COLUMNS):
        for rep in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(X_perm.shape[0])
            X_perm[:, :, i] = X_perm[perm_idx, :, i]
            r2_perm, _, _ = batch_evaluate(model, X_perm, y, y_mean, y_std, device)
            r2_drop[i, rep] = baseline_r2 - r2_perm

    records = []
    for i, col in enumerate(FEATURE_COLUMNS):
        records.append({
            "feature": col,
            "group": FEAT_TO_GROUP.get(col, "Other"),
            "importance_mean": float(r2_drop[i].mean()),
            "importance_std":  float(r2_drop[i].std(ddof=1)) if n_repeats > 1 else 0.0,
        })
    df = pd.DataFrame(records)
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def group_permutation_importance(model, X, y, y_mean, y_std,
                                 n_repeats=5, device="cpu", seed=42):
    """
    特征组级别的 Permutation Importance。
    每组所有特征同时 shuffle，测 R² 下降。
    """
    rng = np.random.default_rng(seed)
    baseline_r2, _ = get_baseline(model, X, y, y_mean, y_std, device)

    # Build index mapping for groups
    group_to_indices = {}
    for gname, feats in FEATURE_GROUPS.items():
        group_to_indices[gname] = [FEATURE_COLUMNS.index(f) for f in feats]

    records = []
    for gname, indices in group_to_indices.items():
        drops = []
        for rep in range(n_repeats):
            X_perm = X.copy()
            for idx in indices:
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, :, idx] = X_perm[perm_idx, :, idx]
            r2_perm, _, _ = batch_evaluate(model, X_perm, y, y_mean, y_std, device)
            drops.append(baseline_r2 - r2_perm)
        records.append({
            "group": gname,
            "n_features": len(indices),
            "features": ", ".join(FEATURE_GROUPS[gname]),
            "importance_mean": float(np.mean(drops)),
            "importance_std":  float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
        })
        print(f"  Group '{gname}' ({len(indices)} feats): "
              f"R² drop = {float(np.mean(drops)):.4f} ± {float(np.std(drops, ddof=1)):.4f}")

    df = pd.DataFrame(records)
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────
# 一站式运行入口
# ──────────────────────────────────────────────────────

def run_phase1(X_test, y_test):
    """Phase 1: 特征-ΔSoC 相关性分析"""
    print("\n" + "=" * 60)
    print("  Phase 1: 特征-ΔSoC 相关性分析")
    print("=" * 60)
    corr_df, pooled = compute_feature_target_corr(X_test, y_test)
    corr_df_sorted = corr_df.sort_values("spearman_r", ascending=False)
    print("\n特征 vs ΔSoC (Spearman 相关性排序):")
    for _, r in corr_df_sorted.iterrows():
        print(f"  {r['feature']:30s}  Spearman={r['spearman_r']:+.4f}  "
              f"Pearson={r['pearson_r']:+.4f}  p={r['spearman_p']:.2e}")
    return corr_df, corr_df_sorted, pooled


def run_phase2(X_test, y_test, y_mean, y_std, device="cpu"):
    """Phase 2: Permutation Importance"""
    print("\n" + "=" * 60)
    print("  Phase 2: Permutation Importance (逐特征)")
    print("=" * 60)
    model = load_model(device)
    imp_df = permutation_importance(model, X_test, y_test, y_mean, y_std,
                                    n_repeats=5, device=device)
    print("\n特征重要性排序 (R² drop):")
    for _, r in imp_df.iterrows():
        bar = "█" * max(1, int(r["importance_mean"] * 100))
        print(f"  {r['feature']:30s}  ΔR²={r['importance_mean']:.4f} ± {r['importance_std']:.4f}  {bar}")
    return model, imp_df


def run_phase3(model, X_test, y_test, y_mean, y_std, device="cpu"):
    """Phase 3: 特征组贡献"""
    print("\n" + "=" * 60)
    print("  Phase 3: 特征组贡献 (Group Permutation)")
    print("=" * 60)
    gdf = group_permutation_importance(model, X_test, y_test, y_mean, y_std,
                                       n_repeats=5, device=device)
    print("\n特征组重要性排序:")
    for _, r in gdf.iterrows():
        bar = "█" * max(1, int(r["importance_mean"] * 50))
        print(f"  {r['group']:20s} ({r['n_features']} feats): "
              f"ΔR²={r['importance_mean']:.4f} ± {r['importance_std']:.4f}  {bar}")
    return gdf


def main():
    device = "cpu"
    print(f"Device: {device}")
    X_test, y_test, y_mean, y_std = load_test_data()

    # Phase 1
    corr_df, corr_df_sorted, pooled = run_phase1(X_test, y_test)

    # Phase 2 + 3 (need model)
    model, imp_df = run_phase2(X_test, y_test, y_mean, y_std, device)
    gdf = run_phase3(model, X_test, y_test, y_mean, y_std, device)

    print("\n" + "=" * 60)
    print("  特征影响分析完成！运行 plot_feature_analysis.py 生成图表。")
    print("=" * 60)

    return {"corr_df": corr_df, "imp_df": imp_df, "gdf": gdf,
            "X_test": X_test, "y_test": y_test, "pooled": pooled,
            "y_mean": y_mean, "y_std": y_std, "model": model}


if __name__ == "__main__":
    main()
