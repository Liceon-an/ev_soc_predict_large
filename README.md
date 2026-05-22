# EV SoC 预测 — 电动车辆荷电状态预测

## 项目概述

通过时间序列数据预测电动车辆在给定窗口内的 **SoC 变化量 (ΔSoC)**。

- **当前服务器**: `ssh -p 12395 root@connect.bjb2.seetacloud.com` (密码 `OKnKgaAc0jYJ`)
- **项目路径**: `/root/code/ev_soc_predict`
- **GPU 环境**: `conda activate torch_gpu` (Python 3.10 + torch 2.11.0+cu130, RTX 5090)

---

## 项目结构

```
ev_soc_predict/
├── configs/                        # 配置文件
│   ├── config_200s.py ~ 1500s.py   # 7 窗口的时间片段配置
│   ├── model_config.py             # 模型超参数 (LSTM/Transformer)
│   ├── train_config.py             # 训练超参数
│   └── path_config.py              # 路径管理
├── src/
│   ├── main.py                     # 训练入口 (支持 --window --model)
│   ├── dataset.py                  # PyTorch Dataset + DataLoader
│   ├── trainer.py                  # 训练循环 (早停/检查点/L1 Loss)
│   ├── utils.py                    # 评估工具 (MAPE/分段分析)
│   ├── models/
│   │   ├── __init__.py             # 模型注册
│   │   ├── lstm_baseline.py        # 纯 LSTM (546K params) ★推荐
│   │   ├── lstm_transformer.py     # LSTM→Transformer (1.17M)
│   │   └── mlr_baseline.py         # 多元线性回归 (36 params)
│   ├── dataprocess/
│   │   ├── compute_feature.py      # 特征工程: aligned CSV → 35 维特征
│   │   ├── time_segment.py         # 滑动窗口切分: feature_data → 片段索引
│   │   ├── build_dataset.py        # 数据集构建: 索引 → train/val/test .npz
│   │   └── process_full_data.py    # 一键: 运行完整 time_segment 流水线
│   └── analysis/                   # 可视化与分析
│       ├── run_all.py              # 运行所有静态图 (无需GPU)
│       ├── plot_utils.py           # 共享常量/数据/颜色
│       ├── plot_01_r2.py           # R² 趋势图
│       ├── plot_02_metrics.py      # MAE/RMSE/MAPE 四指标对比
│       ├── plot_04_lstm_vs_linear.py  # LSTM vs Linear 对比
│       ├── plot_05_correlation.py  # avg_power vs ΔSoC 相关性
│       ├── plot_06_scatter.py      # 预测 vs 真实散点图 (需GPU)
│       ├── plot_07_residuals.py    # 残差分布图 (需GPU)
│       ├── plot_10_chain_concat.py # 多段拼接链式预测 (需GPU)
│       ├── plot_feature_analysis.py# 特征影响分析可视化
│       ├── feature_analysis.py     # 特征影响分析核心计算
│       ├── plot_speed_kmeans.py    # 速度聚类可视化
│       ├── plot_16_feature_pie.py   # 特征贡献饼图
│       ├── plot_17_feature_bar.py   # 特征重要性柱状图
│       └── speed_kmeans.py         # 速度 KMeans 聚类
├── data/
│   ├── raw/                        # 原始 CSV (Dataset1/2, evd.csv)
│   ├── aligned/                    # 对齐后数据 (aligned_data_refined_soc.csv)
│   ├── processed/                  # feature_data.csv (123785×35)
│   ├── split/                      # 时间片段索引 (origin_{N}s.npz × 7)
│   └── train/                      # 训练数据集 (split_{N}s/ × 7)
├── models/                         # 21 个已训练模型 (3 模型 × 7 窗口)
├── plots/                          # 可视化产出 (40+ 张图)
└── logs/                           # 训练日志 + 三模型对比结果
```

---

## 完整数据流水线

```
┌──────────────────────────────────────────────────────────────────────┐
│  阶段 1: 特征工程                                                     │
│  data/aligned/aligned_data_refined_soc.csv                           │
│       │                                                              │
│       ▼                                                              │
│  compute_feature.py  ──────────────────────────────────────────────► │
│  输出: data/processed/feature_data.csv (123785 行 × 35 列)           │
│  功能: 加载→预处理→差值计算→乘积→滑动窗口统计→占比→整合               │
├──────────────────────────────────────────────────────────────────────┤
│  阶段 2: 时间片段划分                                                 │
│  data/processed/feature_data.csv                                     │
│       │                                                              │
│       ▼                                                              │
│  time_segment.py --window {N}  ────────────────────────────────────► │
│  输出: data/split/origin_{N}s.npz (片段索引, shape: [M, T])           │
│  功能: 滑动窗口遍历时间序列，检测断点(>窗口+100s)，生成片段索引        │
│  参数: CORE_STEPS=N/10, STEP_SIZE=N/20, BREAK_THRESHOLD=N+100s       │
├──────────────────────────────────────────────────────────────────────┤
│  阶段 3: 数据集构建                                                   │
│  feature_data.csv + origin_{N}s.npz                                  │
│       │                                                              │
│       ▼                                                              │
│  build_dataset.py --window {N}  ───────────────────────────────────► │
│  输出: data/train/split_{N}s/{train,val,test}_data.npz + scaler.npz │
│  功能:                                                                 │
│    1. 选取 17 维特征 (FEATURE_COLUMNS)                                │
│    2. 计算标签: y_main=ΔSoC(refined_soc[0]-refined_soc[-1])          │
│                y_aux=总能量(power.sum())                              │
│    3. 丢弃首尾各 2 个片段 (N_DROP_FIRST/LAST)                        │
│    4. 随机打散 (seed=42) → 按 70/15/15 划分                          │
│    5. Z-Score 标准化 (RAW_FEATURES 跳过, CLIP_FEATURES 裁剪)          │
├──────────────────────────────────────────────────────────────────────┤
│  阶段 4: 模型训练                                                     │
│  train/val/test_data.npz + scaler.npz                                │
│       │                                                              │
│       ▼                                                              │
│  main.py --window {N} --model {lstm|lstm_transformer|mlr}  ────────► │
│  输出: models/best_model_{N}s_{model}.pt + logs/train_*.log          │
│  功能: 加载数据→构建模型→训练(AdamW+L1+早停)→测试评估                 │
├──────────────────────────────────────────────────────────────────────┤
│  阶段 5: 可视化与分析                                                 │
│  模型 .pt 文件 + 评估指标                                             │
│       │                                                              │
│       ▼                                                              │
│  run_all.py                    → 4 张静态图 (无需 GPU)                │
│  plot_06_scatter.py            → 预测 vs 真实散点图 (需 GPU)          │
│  plot_07_residuals.py          → 残差分布图 (需 GPU)                  │
│  plot_10_chain_concat.py       → 链式预测 6 张图 (需 GPU)             │
│  plot_feature_analysis.py      → 特征分析 5 张图 (需 GPU)             │
│  plot_16_feature_pie.py        → 特征贡献饼图                         │
│  plot_17_feature_bar.py        → 特征重要性柱状图                     │
│  plot_speed_kmeans.py          → 速度聚类 2 张图                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 特征工程详解

### 输入 → 输出

| 步骤 | 脚本 | 输入 | 输出 | 形状 |
|:-----|:-----|:-----|:-----|:-----|
| 1 | `compute_feature.py` | `aligned_data_refined_soc.csv` | `feature_data.csv` | 123785 × 35 |
| 2 | `time_segment.py` | `feature_data.csv` | `origin_{N}s.npz` | M × T (T=N/10) |
| 3 | `build_dataset.py` | 上述 2 个文件 | `{split}_data.npz` | N × T × 17 |

### compute_feature.py 处理模块

1. **数据加载**: 读取 aligned CSV
2. **预处理**: 时间解析、排序、缺失值处理
3. **判断模块**: 标记 `is_run`(行驶状态)、`is_new_trip`(新行程)
4. **差值计算**: speed_diff, mileage_diff, time_diff
5. **特征乘积**: power = total_volt × total_current
6. **滑动窗口统计**: 各特征的 window20 mean/std
7. **占比计算**: cruising_ratio (巡航比例), Low/Mid/High 驾驶模式
8. **整合输出**: 35 列特征数据

### build_dataset.py 选取的 17 维特征

```
speed, speed_diff, mileage_diff,
speed_window20_mean, speed_diff_window20_mean,
temperature_c_window20_mean, relative_humidity_window20_mean,
visibility_km_window20_mean, wind_speed_ms_window20_mean,
speed_window20_std,
Low, Mid, High, cruising_ratio,
total_volt, total_current, power
```

> **特征贡献度**: power + total_current > 76%

### 归一化策略

| 类型 | 特征 | 处理方式 |
|:-----|:-----|:-----|
| **RAW_FEATURES** | Low, Mid, High, cruising_ratio | 跳过 Z-Score，保留 0~1 原始值 |
| **CLIP_FEATURES** | mileage_diff | Z-Score → clip(-5, 5) |
| **普通特征** | 其余 12 个 | Z-Score 标准化 |

---

## 模型架构

### 三模型对比

| 模型 | 文件 | 参数量 | 架构 |
|:-----|:-----|:------:|:-----|
| **LSTM** (推荐) | `src/models/lstm_baseline.py` | 546K | LSTM(2层,128,双向) → Pool → Linear heads |
| LSTM-Transformer | `src/models/lstm_transformer.py` | 1.17M | LSTM → Proj → Transformer(3层,8头) → Pool → Heads |
| MLR | `src/models/mlr_baseline.py` | 36 | MeanPool(T) → Linear(17→2) |

### 模型 1: 纯 LSTM (推荐)

```
Input: (B, T, 17)
  → LSTM(input=17, hidden=128, layers=2, bi=True, dropout=0.2)
  → (B, T, 256)
  → Global Mean Pooling 或 Last Step
  → main_head(256→1): ΔSoC 预测
  → aux_head(256→1):  总能量 (仅监控)
总参数: ~546K (不随窗口变化)
```

### 模型 2: LSTM-Transformer

```
Input: (B, T, 17)
  → LSTM(input=17, hidden=128, layers=2, bi=True, dropout=0.2)
  → (B, T, 256)
  → Linear(256→128) + LayerNorm(128) + Dropout(0.1)
  → PositionalEncoding(d_model=128)
  → TransformerEncoder × 3 (nhead=8, FFN=512, GELU)
  → Global Mean Pooling
  → main_head(128→1): ΔSoC 预测
  → aux_head(128→1):  总能量 (仅监控)
总参数: ~1.17M (不随窗口变化)
```

### 模型 3: MLR (多元线性回归)

```
Input: (B, T, 17)
  → Global Mean Pooling → (B, 17)
  → main_head(17→1): ΔSoC 预测
  → aux_head(17→1):  总能量 (仅监控)
总参数: 36
```

### 模型超参数 (model_config.py)

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| `lstm_input_dim` | 17 | 输入特征维度 |
| `lstm_hidden_dim` | 128 | LSTM 隐藏层大小 |
| `lstm_num_layers` | 2 | LSTM 层数 |
| `lstm_dropout` | 0.2 | LSTM dropout |
| `lstm_bidirectional` | True | 双向 LSTM |
| `d_model` | 128 | Transformer 维度 |
| `nhead` | 8 | 注意力头数 |
| `transformer_num_layers` | 3 | Transformer 层数 |
| `transformer_dim_feedforward` | 512 | FFN 维度 |
| `transformer_dropout` | 0.1 | Transformer dropout |
| `pooling` | `"mean"` | 池化方式 (mean/last) |

---

## 训练超参数 (train_config.py)

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| `seed` | 42 | 随机种子 |
| `epochs` | 200 | 最大训练轮数 |
| `batch_size` | 64 | 批次大小 |
| `lr` | 1e-3 | 初始学习率 |
| `weight_decay` | 1e-5 | 权重衰减 (L2) |
| `lr_scheduler_factor` | 0.5 | LR 衰减因子 |
| `lr_scheduler_patience` | 5 | LR 衰减等待轮数 |
| `early_stop_patience` | 20 | 早停等待轮数 |
| `grad_clip_norm` | 10.0 | 梯度裁剪阈值 |
| **Loss** | **L1Loss (MAE)** | 经 Huber 消融验证 L1 最优 |
| **优化器** | **AdamW** | — |
| **调度器** | **ReduceLROnPlateau(mode=min)** | — |

---

## 窗口配置

每个窗口有独立配置文件，定义时间片段参数：

| 窗口 | CORE_STEPS | TIME_RESOLUTION | 步长 | 训练样本 |
|:-----|:----------:|:---------------:|:----:|:--------:|
| 200s | 20 | 10s | 10 | 7,782 |
| 400s | 40 | 10s | 20 | 3,496 |
| 600s | 60 | 10s | 30 | 2,117 |
| 800s | 80 | 10s | 40 | 1,453 |
| 1000s | 100 | 10s | 50 | 1,066 |
| 1200s | 120 | 10s | 60 | 812 |
| 1500s | 150 | 10s | 75 | 574 |

> 所有窗口的时间分辨率统一为 10 秒/步

---

## 快速命令

```bash
# ========== 连接服务器 ==========
ssh -p 12395 root@connect.bjb2.seetacloud.com   # 密码: OKnKgaAc0jYJ
cd /root/code/ev_soc_predict

# ========== 激活 GPU 环境 (重要!) ==========
source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu

# ========== 数据流水线 (以 800s 为例) ==========
python3 src/dataprocess/time_segment.py --window 800
# -> 输出: data/split/origin_800s.npz

python3 src/dataprocess/build_dataset.py --window 800
# -> 输出: data/train/split_800s/{train,val,test}_data.npz + scaler.npz

# ========== 模型训练 ==========
# 纯 LSTM (推荐, 546K 参数)
python3 src/main.py --window 800 --model lstm
# -> 输出: models/best_model_800s_lstm.pt + logs/train_*.log

# LSTM-Transformer (1.17M 参数)
python3 src/main.py --window 800 --model lstm_transformer
# -> 输出: models/best_model_800s_lstm_transformer.pt + logs/train_*.log

# MLR 线性基线 (36 参数)
python3 src/main.py --window 800 --model mlr
# -> 输出: models/best_model_800s_mlr.pt + logs/train_*.log

# ========== 可视化 ==========
# 静态图 (无需 GPU, 约 30 秒)
python3 src/analysis/run_all.py
# -> 输出: plots/01_r2_trend.png, 02_metrics_comparison.png,
#          plots/04_lstm_vs_linear.png, 05_correlation_trend.png

# 预测散点图 + 残差分布 (需 GPU, 7 窗口推理)
python3 src/analysis/plot_06_scatter.py
# -> 输出: plots/06_scatter_comparison.png

python3 src/analysis/plot_07_residuals.py
# -> 输出: plots/07_residual_dist.png

# 多段拼接链式预测 (需 GPU)
python3 src/analysis/plot_10_chain_concat.py
# -> 输出: plots/10a_full_timeline.png ~ 10f_error_accumulation.png

# 特征影响分析 (需 GPU)
python3 src/analysis/plot_feature_analysis.py
# -> 输出: plots/12_feature_corr_heatmap.png ~ 15_summary_dashboard.png

# 速度聚类分析
python3 src/analysis/plot_speed_kmeans.py
# -> 输出: plots/11_speed_kmeans.png, 11_speed_kmeans_box.png

# 特征贡献图
python3 src/analysis/plot_16_feature_pie.py
# -> 输出: plots/16_feature_contribution_pie.png

python3 src/analysis/plot_17_feature_bar.py
# -> 输出: plots/17_feature_bar.png

# ========== 查看结果 ==========
cat logs/three_model_comparison.txt          # 三模型 × 七窗口完整对比
ls -la plots/                               # 所有可视化图片
ls models/                                  # 所有模型文件 (21 个)
tail -30 logs/train_*.log                   # 最新训练日志
```

---

## 核心实验结果

### 三模型 × 七窗口完整对比

| Window | Model | R² | MAE | RMSE | MAPE | Params |
|:------:|:------|:---:|:----:|:----:|:----:|:------:|
| **200s** | MLR | 0.7570 | 0.0952 | 0.1728 | 23.96% | 36 |
| | LSTM-Transformer | 0.7885 | 0.0853 | 0.1612 | 22.26% | 1.17M |
| | **LSTM** | **0.7846** | **0.0842** | **0.1627** | **20.71%** | 546K |
| **400s** | MLR | 0.8279 | 0.1459 | 0.2505 | 23.79% | 36 |
| | LSTM-Transformer | 0.8245 | 0.1374 | 0.2530 | 22.09% | 1.17M |
| | **LSTM** | **0.8596** | **0.1278** | **0.2263** | **20.75%** | 546K |
| **600s** | MLR | 0.8769 | 0.1780 | 0.2941 | 23.34% | 36 |
| | LSTM-Transformer | 0.8618 | 0.1716 | 0.3117 | 21.08% | 1.17M |
| | **LSTM** | **0.8969** | **0.1595** | **0.2691** | **18.96%** | 546K |
| **800s** | MLR | 0.8873 | 0.2100 | 0.3280 | 21.86% | 36 |
| | LSTM-Transformer | **0.9204** | 0.1858 | 0.2756 | 18.68% | 1.17M |
| | LSTM | 0.9175 | **0.1764** | 0.2806 | **15.19%** | 546K |
| **1000s** | MLR | 0.8961 | 0.2422 | 0.3747 | 20.02% | 36 |
| | LSTM-Transformer | 0.9154 | 0.2230 | 0.3382 | 16.68% | 1.17M |
| | **LSTM** | **0.9241** | **0.2114** | **0.3204** | **16.33%** | 546K |
| **1200s** | MLR | 0.9218 | 0.2754 | 0.3925 | 26.45% | 36 |
| | LSTM-Transformer | 0.9247 | 0.2614 | 0.3852 | 19.14% | 1.17M |
| | **LSTM** | **0.9373** | **0.2367** | **0.3515** | **18.78%** | 546K |
| **1500s** | MLR | 0.9250 | 0.3030 | 0.4044 | 39.19% | 36 |
| | LSTM-Transformer | 0.9467 | 0.2262 | 0.3410 | 14.46% | 1.17M |
| | **LSTM** | **0.9624** | **0.1847** | **0.2863** | **13.01%** | 546K |

> 完整对比表存档: `logs/three_model_comparison.txt`

### 核心发现

1. **LSTM-Transformer 优于 LSTM** — Transformer 编码器提升预测能力
2. **MLR (36 参数) 在短窗口接近深度学习** — 400s R²=0.8279 甚至超过 LSTM-Transformer (0.8245)
3. **窗口越长，深度学习优势越大** — LSTM vs MLR ΔR²: 200s +0.028 → 1500s +0.037
4. **800s 是最佳落地窗口** — LSTM 800s MAPE=15.19% 全窗口最低，数据覆盖率 79.5%
5. **R² 单调递增趋势在所有模型上复现** — 物理信号增强是稳健现象
6. **MLR MAPE 在 1500s 爆炸 (39.19%)** — 线性模型无法处理大 ΔSoC 场景

---

## 可视化产出清单

### 静态图 (无需 GPU, run_all.py)

| 文件 | 说明 |
|:-----|:-----|
| `01_r2_trend.png` | R² 跨窗口趋势 (三模型) |
| `02a_r2_mape.png` | R² + MAPE 双轴图 |
| `02b_mae_rmse.png` | MAE + RMSE 对比 |
| `02c_norm_mae_rmse.png` | 归一化 MAE/RMSE |
| `04_lstm_vs_linear.png` | LSTM vs Linear (R² + MAE) |
| `05_correlation_trend.png` | avg_power vs ΔSoC 相关性趋势 |

### GPU 推理图

| 文件 | 说明 |
|:-----|:-----|
| `06_scatter_comparison.png` | 7 窗口预测 vs 真实散点 |
| `07_residual_dist.png` | 7 窗口残差分布 |

### 链式预测图 (plot_10_chain_concat.py)

| 文件 | 说明 |
|:-----|:-----|
| `10a_full_timeline.png` | 12.9h 全时间线 SoC 轨迹 |
| `10b_zoom_3h.png` | 前 3h 放大 |
| `10c_error_timeline.png` | 每步预测误差时序 |
| `10d_scatter.png` | 预测 vs 真实散点 |
| `10e_metrics_by_trip.png` | 各段行程误差柱状 |
| `10f_error_accumulation.png` | 累积误差增长 |

### 特征分析图

| 文件 | 说明 |
|:-----|:-----|
| `12_feature_corr_heatmap.png` | 17 特征相关热力图 |
| `16_feature_contribution_pie.png` | 特征贡献饼图 |
| `17_feature_bar.png` | 特征重要性柱状图 |

### 速度分析图 (plot_speed_kmeans.py)

| 文件 | 说明 |
|:-----|:-----|
| `11_speed_kmeans.png` | KMeans 速度聚类直方图 |
| `11_speed_kmeans_box.png` | 各簇速度箱线图 |

---

## 链式预测

模型预测单个窗口的 ΔSoC。链式预测在完整行程上滑动窗口，逐段累加 ΔSoC 重建 SoC 曲线。

### 结果 (Trip #1159, S1500 模型)

| 策略 | Stride | 预测点 | MAE | 说明 |
|:-----|:------:|:------:|:----:|:-----|
| 全窗口无重叠 | 150 | 9 | **0.2885** | 每 25min 更新，精度最高 |
| 半重叠 | 75 | 18 | 0.7460 | ΔSoC 按 50% 比例分配 |
| 细粒度 | 15 | 87 | 1.1400 | 线性假设误差累积 |

### 多段拼接 (7 trips × 12.9h, SoC 100%→36%)

| 全局 MAE | 全局 ΔSoC | 相对误差 |
|:--------:|:---------:|:--------:|
| 0.37 | 63.9% | **0.58%** |

---

## 环境说明

| 项目 | 系统默认 | torch_gpu (conda) |
|:-----|:---------|:------------------|
| Python | 3.8.10 | 3.10.20 |
| PyTorch | 2.4.1+cu121 | 2.11.0+cu130 |
| CUDA | 12.1 | 13.0 |
| GPU | 不可用 (sm_120 不兼容) | RTX 5090 可用 |

> 运行 GPU 脚本前必须: `source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu`

---

## 后续计划

### P0 — 模型落地
- [ ] S800 vs S1500 链式预测对比 (确定落地用哪个窗口)
- [ ] LSTM 组件消融 (双向/单向、层数、hidden dim、pooling)

### P1 — 精度提升
- [ ] GRU 对比
- [ ] 特征筛选 (Top-N)
- [ ] 特征扩展 (电池温度、内阻、充放电状态)

### P2 — 工程
- [ ] 三模型对比可视化更新
- [ ] TensorBoard/W&B 集成

---

## 版本

**版本**: 2.0 | **更新日期**: 2026-05-18 | **项目**: ev_soc_predict
