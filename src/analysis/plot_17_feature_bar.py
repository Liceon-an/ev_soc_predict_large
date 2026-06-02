"""
Plot 17: 单特征贡献度 — 水平柱状图。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 先加载业务库
from src.analysis.feature_analysis import run_phase2, load_test_data

# ====================== 【中文映射表】======================
FEATURE_CN = {
    "speed": "车速",
    "speed_diff": "车速差分",
    "mileage_diff": "里程差",
    "speed_window20_mean": "车速窗口均值",
    "speed_diff_window20_mean": "车速差分窗口均值",
    "temperature_c_window20_mean": "温度窗口均值",
    "relative_humidity_window20_mean": "相对湿度窗口均值",
    "visibility_km_window20_mean": "能见度窗口均值",
    "wind_speed_ms_window20_mean": "风速窗口均值",
    "speed_window20_std": "车速窗口标准差",
    "Low": "低负载",
    "Mid": "中负载",
    "High": "高负载",
    "cruising_ratio": "巡航比例",
    "total_volt": "总电压",
    "total_current": "总电流",
    "power": "电功率",
}

# ====================== 中文稳定配置 ======================
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"

# ====================== 画布(14,9) 标准专业字号 ======================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "legend.fontsize": 14
})

# ====================== 绘图逻辑 ======================
device = "cpu"
X_test, y_test, y_mean, y_std = load_test_data()
_, imp_df = run_phase2(X_test, y_test, y_mean, y_std, device)

all_features = imp_df["feature"].tolist()
all_imp = imp_df["importance_mean"].tolist()
all_std = imp_df["importance_std"].tolist()
total = sum(v for v in all_imp if v > 0)
threshold = 0.005 * total

top_features, top_values, top_errors = [], [], []
others_sum, others_std_sq, others_count = 0.0, 0.0, 0

for f, v, s in zip(all_features, all_imp, all_std):
    if v >= threshold and v > 0:
        top_features.append(f); top_values.append(v); top_errors.append(s)
    elif v > 0:
        others_sum += v; others_std_sq += s**2; others_count += 1

if others_count > 0:
    top_features.append(f"其他（{others_count} 个特征）")
    top_values.append(others_sum)
    top_errors.append(float(np.sqrt(others_std_sq)))

top_features.reverse(); top_values.reverse(); top_errors.reverse()

# 英文名转中文
top_features_cn = []
for name in top_features:
    if "其他" in name:
        top_features_cn.append(name)
    else:
        top_features_cn.append(FEATURE_CN.get(name, name))

# 画布
fig, ax = plt.subplots(figsize=(14, 9))
n = len(top_features)
# 统一全部柱子为蓝色渐变，去除单独灰色赋值
blues = plt.cm.Blues(np.linspace(0.3, 0.9, n))

bars = ax.barh(range(n), top_values, color=blues, edgecolor="white", height=0.65)
ax.set_yticks(range(n))

ax.set_yticklabels(top_features_cn, fontsize=20)
ax.set_xlabel("R² 下降", fontsize=24)
ax.set_title("特征贡献度", fontsize=28, fontweight="bold")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.invert_yaxis()
ax.tick_params(labelsize=20)

# 数值标签
for b, v in zip(bars, top_values):
    ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
            f"{v:.4f}  ({v/total*100:.1f}%)", va="center", fontsize=18, fontweight="bold")

# 右下角备注
ax.text(0.98, 0.02, f"基线 R² = 0.9462",
        transform=ax.transAxes, ha="right", fontsize=18, color="gray")

plt.tight_layout()
outdir = _PROJECT_ROOT / "plots"
outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "17_feature_bar.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  [17] 单特征贡献度柱状图已保存")

print("\n特征贡献度明细:")
for f, v in zip(all_features, all_imp):
    pct = v / total * 100 if v > 0 else 0
    flag = " <<<" if v >= threshold else ""
    print(f"  {f:35s}  ΔR²={v:+.4f}  {pct:5.1f}%{flag}")