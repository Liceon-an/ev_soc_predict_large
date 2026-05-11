#!/usr/bin/env python3
"""
特征影响分析 — 可视化模块

生成图表:
  12_feature_corr_heatmap.png    — 特征间相关热力图
  12_feature_target_corr.png     — 各特征与 ΔSoC 相关性排序
  12_top_features_scatter.png    — Top 6 特征 vs ΔSoC 散点图
  13_permutation_importance.png  — Permutation Importance 排序
  14_group_importance.png        — 特征组贡献排序
  15_summary_dashboard.png       — 综合仪表盘
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.cluster import hierarchy as hc

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from feature_analysis import (
    FEATURE_COLUMNS, FEATURE_GROUPS, FEAT_TO_GROUP,
    load_test_data, load_model,
    run_phase1, run_phase2, run_phase3,
    compute_feature_corr_matrix, _aggregate_features,
    OUTPUT_DIR,
)

# ─── Style ─────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 14,
    "axes.labelsize": 12, "figure.figsize": (10, 6),
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# 特征组调色板 (与 FEATURE_GROUPS 顺序一致)
GROUP_PALETTE = {
    "Speed (瞬时)":   "#2196F3",
    "Speed (窗口)":   "#00BCD4",
    "Weather":        "#4CAF50",
    "Driving Mode":   "#FF9800",
    "Electrical":     "#F44336",
    "Mileage":        "#9C27B0",
}
FEATURE_PALETTE = [GROUP_PALETTE.get(FEAT_TO_GROUP.get(f, "Other"), "#607D8B")
                   for f in FEATURE_COLUMNS]


# ════════════════════════════════════════════════════════
# Phase 1: 特征-ΔSoC 相关性分析
# ════════════════════════════════════════════════════════

def plot_corr_heatmap(corr_matrix, feature_names, output_path):
    """
    图1: 特征间 Spearman 相关热力图 (带层次聚类).
    目的: 识别特征间的冗余性/共线性.
    解读: 颜色越深 |r| 越大；对角线=1；红色块=高度相关的特征组.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # 层次聚类重排
    linkage = hc.linkage(corr_matrix, method="average")
    order = hc.leaves_list(linkage)
    ordered_names = [feature_names[i] for i in order]
    ordered_matrix = corr_matrix[order][:, order]

    mask = np.triu(np.ones_like(ordered_matrix, dtype=bool), k=1)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(ordered_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5,
                xticklabels=ordered_names, yticklabels=ordered_names,
                cbar_kws={"shrink": 0.8, "label": "Spearman r"},
                ax=ax)

    ax.set_title("Feature-Feature Spearman Correlation (Clustered)", fontsize=14)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Feature correlation heatmap -> {output_path}")


def plot_feature_target_corr(corr_df_sorted, output_path):
    """
    图2: 各特征与 ΔSoC 的 Spearman 相关系数排序.
    目的: 快速识别与预测目标最相关/最不相关的特征.
    解读: 正值 → 特征增大时 ΔSoC 增大；负值 → 相反.
          |r| > 0.3 通常表示有意义的线性关联.
    """
    df = corr_df_sorted.copy()
    colors = [GROUP_PALETTE.get(g, "#607D8B") for g in df["group"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(df)), df["spearman_r"].values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    # 标注显著性
    for i, (_, row) in enumerate(df.iterrows()):
        p = row["spearman_p"]
        label = f"  {row['spearman_r']:.3f}"
        if p < 0.001:
            label += "***"
        elif p < 0.01:
            label += "**"
        elif p < 0.05:
            label += "*"
        ax.text(row["spearman_r"], i, label, va="center",
                fontsize=8, color="black" if abs(row["spearman_r"]) < 0.4 else "white")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman Correlation with ΔSoC", fontsize=12)
    ax.set_title("Feature vs ΔSoC — Spearman Correlation", fontsize=14)

    # 图例
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=g)
                    for g, c in GROUP_PALETTE.items()
                    if g in df["group"].values]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right",
              title="Feature Group", title_fontsize=9)

    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Feature-target correlation -> {output_path}")


def plot_top_features_scatter(pooled, y_test, corr_df_sorted, output_path):
    """
    图3: Top 6 特征 vs ΔSoC 散点图 + 回归线.
    目的: 直观展示相关性最强的特征与目标的关系形态（线性/非线性/异常点）.
    解读: 每个点 = 一个测试样本；蓝线 = 线性拟合趋势；
          观察是否单调、是否有离群簇、是否有天花板/地板效应.
    """
    top6 = corr_df_sorted.head(6)["feature"].tolist()
    idxs = [FEATURE_COLUMNS.index(f) for f in top6]
    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
    axes = axes.flatten()

    for ax_idx, (feat_name, feat_idx) in enumerate(zip(top6, idxs)):
        ax = axes[ax_idx]
        x = pooled[:, feat_idx]
        g = FEAT_TO_GROUP.get(feat_name, "Other")
        c = GROUP_PALETTE.get(g, "#607D8B")

        ax.scatter(x, y_test, alpha=0.4, s=12, color=c, edgecolors="none")
        # 局部加权回归趋势线 (lowess approximation via polyfit)
        try:
            from numpy.polynomial import polynomial as P
            coefs = P.polyfit(x, y_test, 1)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, P.polyval(x_sorted, coefs), color="#E91E63",
                    linewidth=2, linestyle="--", label=f"Linear fit")
        except Exception:
            pass
        ax.set_xlabel(f"{feat_name}", fontsize=10)
        ax.set_ylabel("ΔSoC", fontsize=10)
        ax.set_title(f"{feat_name} vs ΔSoC", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Top 6 Features vs ΔSoC (with Linear Trend)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Top features scatter -> {output_path}")


# ════════════════════════════════════════════════════════
# Phase 2: Permutation Importance
# ════════════════════════════════════════════════════════

def plot_permutation_importance(imp_df, output_path):
    """
    图4: Permutation Importance 排序 (水平条形图).
    目的: 直接衡量每个特征对模型预测精度的边际贡献.
    方法: 逐特征 shuffle → 测 R² 下降量，下降越大越重要.
    解读:
      - ΔR² > 0.01  → 重要特征，显著影响预测
      - ΔR² ~ 0     → 对当前模型贡献很小，可考虑移除
      - 误差棒 = 5 次重复的 ±1σ
    """
    df = imp_df.copy()

    # 转正负值排序
    positive = df[df["importance_mean"] >= 0].sort_values("importance_mean")
    negative = df[df["importance_mean"] < 0].sort_values("importance_mean")
    plot_df = pd.concat([negative, positive], axis=0)

    colors = [GROUP_PALETTE.get(g, "#607D8B") for g in plot_df["group"]]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(plot_df))
    bars = ax.barh(y_pos, plot_df["importance_mean"].values,
                   xerr=plot_df["importance_std"].values,
                   color=colors, edgecolor="white", linewidth=0.5,
                   height=0.7, capsize=3, error_kw={"linewidth": 1.5})

    # 标注值
    for i, (_, row) in enumerate(plot_df.iterrows()):
        val = row["importance_mean"]
        label = f"  {val:.4f}"
        if val >= 0:
            ax.text(val, i, label, va="center", fontsize=8)
        else:
            ax.text(val, i, label, va="center", fontsize=8, ha="right")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=10)
    ax.set_xlabel("ΔR² (R² drop when feature shuffled)", fontsize=12)
    ax.set_title("Permutation Feature Importance (S1500 Model)", fontsize=14)

    from matplotlib.patches import Patch
    used_groups = plot_df["group"].unique()
    legend_elems = [Patch(facecolor=GROUP_PALETTE.get(g, "#607D8B"), label=g)
                    for g in FEATURE_GROUPS if g in used_groups]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right",
              title="Feature Group", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Permutation importance -> {output_path}")


# ════════════════════════════════════════════════════════
# Phase 3: 特征组贡献
# ════════════════════════════════════════════════════════

def plot_group_importance(gdf, output_path):
    """
    图5: 特征组 Permutation Importance.
    目的: 从更高维度看 "哪类特征" 对模型贡献最大.
    方法: 整组特征同时 shuffle → 测 R² 下降.
    解读:
      - Electrical > Speed > Driving Mode > Weather > Mileage
      - 整组贡献 > 单特征之和 (组内协同效应)
      - 归一化: 组内平均特征贡献 = 组贡献 / n_features
    """
    df = gdf.sort_values("importance_mean")

    colors = [GROUP_PALETTE.get(g, "#607D8B") for g in df["group"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── 左图: 绝对贡献 ──
    ax = axes[0]
    bars = ax.barh(range(len(df)), df["importance_mean"].values,
                   xerr=df["importance_std"].values,
                   color=colors, edgecolor="white", linewidth=0.5,
                   height=0.6, capsize=3, error_kw={"linewidth": 1.5})

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["importance_mean"], i, f"  {row['importance_mean']:.4f}",
                va="center", fontsize=9)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r['group']} ({r['n_features']} feats)"
                        for _, r in df.iterrows()], fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔR² (Group Permutation)", fontsize=11)
    ax.set_title("Absolute Group Importance", fontsize=13)

    # ── 右图: 归一化后（每特征平均贡献） ──
    ax = axes[1]
    df_norm = df.copy()
    df_norm["per_feat"] = df_norm["importance_mean"] / df_norm["n_features"]
    df_norm = df_norm.sort_values("per_feat")

    norm_colors = [GROUP_PALETTE.get(g, "#607D8B") for g in df_norm["group"]]
    ax.barh(range(len(df_norm)), df_norm["per_feat"].values,
            color=norm_colors, edgecolor="white", linewidth=0.5, height=0.6)

    for i, (_, row) in enumerate(df_norm.iterrows()):
        ax.text(row["per_feat"], i, f"  {row['per_feat']:.4f}",
                va="center", fontsize=9)

    ax.set_yticks(range(len(df_norm)))
    ax.set_yticklabels(df_norm["group"].values, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔR² per Feature", fontsize=11)
    ax.set_title("Normalized Group Importance (per Feature)", fontsize=13)

    fig.suptitle("Feature Group Contribution Analysis (S1500)",
                 fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Group importance -> {output_path}")


# ════════════════════════════════════════════════════════
# Phase 4: 综合仪表盘
# ════════════════════════════════════════════════════════

def plot_summary_dashboard(corr_df_sorted, imp_df, gdf, output_path):
    """
    图6: 综合仪表盘 — 四合一视图.
    布局:
      左上: 特征-ΔSoC 相关性 (Top 8)
      右上: Permutation Importance (Top 8)
      左下: 特征组贡献 (绝对)
      右下: 总结表格/文字
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Impact Analysis — Summary Dashboard (S1500)",
                 fontsize=16, y=0.98)

    # ── 左上: 相关性 Top 8 ──
    ax = axes[0, 0]
    top8_corr = corr_df_sorted.head(8)
    colors_c = [GROUP_PALETTE.get(g, "#607D8B") for g in top8_corr["group"]]
    ax.barh(range(len(top8_corr)), top8_corr["spearman_r"].values,
            color=colors_c, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(top8_corr)))
    ax.set_yticklabels(top8_corr["feature"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Spearman r")
    ax.set_title("Top 8: Correlation with ΔSoC", fontsize=12)
    ax.invert_yaxis()

    # ── 右上: Permutation Importance Top 8 ──
    ax = axes[0, 1]
    top8_imp = imp_df.head(8)
    colors_p = [GROUP_PALETTE.get(g, "#607D8B") for g in top8_imp["group"]]
    ax.barh(range(len(top8_imp)), top8_imp["importance_mean"].values,
            xerr=top8_imp["importance_std"].values,
            color=colors_p, edgecolor="white", height=0.6,
            capsize=2, error_kw={"linewidth": 1})
    ax.set_yticks(range(len(top8_imp)))
    ax.set_yticklabels(top8_imp["feature"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔR²")
    ax.set_title("Top 8: Permutation Importance", fontsize=12)
    ax.invert_yaxis()

    # ── 左下: 特征组贡献 ──
    ax = axes[1, 0]
    gdf_sorted = gdf.sort_values("importance_mean")
    colors_g = [GROUP_PALETTE.get(g, "#607D8B") for g in gdf_sorted["group"]]
    ax.barh(range(len(gdf_sorted)), gdf_sorted["importance_mean"].values,
            color=colors_g, edgecolor="white", height=0.5)
    for i, (_, row) in enumerate(gdf_sorted.iterrows()):
        ax.text(row["importance_mean"], i,
                f"  {row['importance_mean']:.3f}  ({row['n_features']} feats)",
                va="center", fontsize=8)
    ax.set_yticks(range(len(gdf_sorted)))
    ax.set_yticklabels(gdf_sorted["group"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔR² (Group Permutation)")
    ax.set_title("Feature Group Contribution", fontsize=12)

    # ── 右下: 结论表格 ──
    ax = axes[1, 1]
    ax.axis("off")

    # 构建总结数据
    key_features = imp_df.head(5)["feature"].tolist()
    low_features = imp_df.tail(3)["feature"].tolist()
    top_group = gdf.iloc[0]["group"]
    baseline_r2 = 0.9462  # S1500 baseline (from experiment)

    summary_text = (
        "═══ 综合结论 ═══\n\n"
        f"基线 S1500 Test R² = {baseline_r2:.4f}\n\n"
        "■ 最关键特征 (Top 5):\n"
        + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(key_features)) +
        "\n\n■ 低贡献特征 (可考虑移除):\n"
        + "\n".join(f"  • {f}" for f in low_features) +
        f"\n\n■ 最重要特征组:\n  {top_group}\n\n"
        "■ 关键结论:\n"
        "  1. 电学特征(power/total_current)\n"
        "     贡献最大，符合物理直觉\n"
        "  2. 速度特征提供重要补充,\n"
        "     窗口统计比瞬时值更稳定\n"
        "  3. 气象特征贡献较弱,\n"
        "     但极端天气下可能更重要\n"
        "  4. 驾驶模式类特征贡献有限,\n"
        "     可能与速度特征冗余"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", va="top",
            linespacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Summary dashboard -> {output_path}")


# ════════════════════════════════════════════════════════
# 开关函数
# ════════════════════════════════════════════════════════

def run_all_plots():
    """运行所有图表生成"""
    print("=" * 60)
    print("  特征影响分析 — 可视化")
    print("=" * 60)

    # 1. 加载数据
    device = "cpu"
    X_test, y_test, y_mean, y_std = load_test_data()

    # 2. Phase 1 计算 (不需要模型)
    corr_df, corr_df_sorted, pooled = run_phase1(X_test, y_test)

    # 3. Phase 1 作图
    print("\n--- Phase 1: 相关性分析作图 ---")
    corr_matrix, _ = compute_feature_corr_matrix(X_test)
    plot_corr_heatmap(corr_matrix, FEATURE_COLUMNS,
                      OUTPUT_DIR / "12_feature_corr_heatmap.png")
    plot_feature_target_corr(corr_df_sorted,
                             OUTPUT_DIR / "12_feature_target_corr.png")
    plot_top_features_scatter(pooled, y_test, corr_df_sorted,
                              OUTPUT_DIR / "12_top_features_scatter.png")

    # 4. 加载模型 + Phase 2/3 计算
    print("\n--- Phase 2 & 3: 加载模型并计算 ---")
    model = load_model(device)
    _, imp_df = run_phase2(X_test, y_test, y_mean, y_std, device)
    gdf = run_phase3(model, X_test, y_test, y_mean, y_std, device)

    # 5. Phase 2/3 作图
    print("\n--- Phase 2 & 3: 作图 ---")
    plot_permutation_importance(imp_df,
                                OUTPUT_DIR / "13_permutation_importance.png")
    plot_group_importance(gdf,
                          OUTPUT_DIR / "14_group_importance.png")

    # 6. Phase 4: 综合仪表盘
    print("\n--- Phase 4: 综合仪表盘 ---")
    plot_summary_dashboard(corr_df_sorted, imp_df, gdf,
                           OUTPUT_DIR / "15_summary_dashboard.png")

    print("\n" + "=" * 60)
    print("  全部图表生成完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    return {"corr_df": corr_df_sorted, "imp_df": imp_df, "gdf": gdf}


if __name__ == "__main__":
    run_all_plots()
