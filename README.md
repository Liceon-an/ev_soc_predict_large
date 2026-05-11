# EV SoC 预测 — 电动车辆荷电状态预测

## 项目概述

通过时间序列数据预测电动车辆在给定窗口内的 **SoC 变化量 (ΔSoC)**。

- **远程服务器**: `ssh -p 47501 root@connect.westd.seetacloud.com`
- **项目路径**: `/root/code/ev_soc_predict`
- **GPU 环境**: `conda activate torch_gpu` (Python 3.10 + torch 2.11.0+cu130)
- **GPU**: NVIDIA RTX 5090 (32GB, Blackwell)

---

## 项目结构

```
ev_soc_predict/
├── configs/                    # 配置文件
│   ├── config_200s.py ~ 1500s.py  # 7 窗口的时间片段配置
│   ├── model_config.py         # 模型超参数
│   ├── train_config.py         # 训练超参数
│   └── path_config.py          # 路径管理
├── src/
│   ├── main.py                 # 训练入口
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   ├── trainer.py              # 训练循环 (早停 / 检查点 / 评估)
│   ├── models/
│   │   └── lstm_transformer.py # 核心模型: LSTM → Transformer
│   ├── dataprocess/
│   │   ├── compute_feature.py  # 特征工程 (→ 35 维)
│   │   ├── time_segment.py     # 时间窗口滑窗切分
│   │   └── build_dataset.py    # 数据集构建 (标准化 / 划分)
│   └── analysis/               # 可视化与分析脚本 (15+)
├── data/
│   ├── raw/                    # 原始数据
│   ├── aligned/                # 对齐后数据
│   ├── processed/              # feature_data.csv (123785×35)
│   ├── split/                  # 时间片段索引 (.npz)
│   └── train/                  # 训练数据集 (split_200s ~ split_1500s)
├── models/                     # 7 窗口已训练模型 (.pt)
├── plots/                      # 可视化产出 (40+ 张图)
└── logs/                       # 训练日志 + 实验记录
```

---

## 数据流水线

```
原始 CSV → [compute_feature.py] → feature_data.csv (123785×35)
         → [time_segment.py]    → origin_{N}s.npz (片段索引)
         → [build_dataset.py]   → train/val/test_data.npz + scaler.npz
         → [main.py]            → 模型训练
```

---

## 模型架构

```
Input (B, T, 17)
  → LSTM (2 层, 双向, hidden=128) → (B, T, 256)
  → Linear + LayerNorm + Dropout → (B, T, 128)
  → PositionalEncoding
  → TransformerEncoder ×3 (nhead=8, FFN=512)
  → Global Mean Pooling → (B, 128)
  → y_main (ΔSoC, 训练目标)
  → y_aux  (总能量, 仅监控)
```

- 总参数量: **~1.17M** (固定，不随窗口变化)
- Loss: L1Loss (MAE)
- 优化器: AdamW (lr=1e-3, wd=1e-5)
- 调度器: ReduceLROnPlateau (patience=5)
- 早停: patience=20, 梯度裁剪: max_norm=10.0

---

## 快速命令

```bash
# 连接远程服务器
ssh -p 47501 root@connect.westd.seetacloud.com

# 激活 GPU 环境
source /root/miniconda3/etc/profile.d/conda.sh && conda activate torch_gpu

# 数据流水线 (以 800s 为例)
python3 src/dataprocess/time_segment.py --window 800
python3 src/dataprocess/build_dataset.py --window 800
python3 src/main.py --window 800

# 可视化
python3 src/analysis/run_all.py                   # 静态图
python3 src/analysis/plot_10_chain_prediction.py  # 链式预测 (需 GPU)
python3 src/analysis/plot_06_scatter.py           # 散点图 (需 GPU)
python3 src/analysis/plot_07_residuals.py         # 残差图 (需 GPU)

# 查看结果
cat logs/experiment_log.csv
ls plots/
```

---

## 7 窗口消融实验结果

| 窗口 | 样本数 | Test R² | Test MAE | MAPE | 备注 |
|:-----|:------:|:-------:|:--------:|:----:|:-----|
| S200 | 7,782 | 0.7885 | 0.0853 | 22.26% | |
| S400 | 3,496 | 0.8245 | 0.1374 | 22.09% | 默认 |
| S600 | 2,117 | 0.8618 | 0.1716 | 21.08% | |
| S800 | 1,453 | 0.9204 | 0.1858 | 18.68% | **性价比拐点** |
| S1000 | 1,066 | 0.9154 | 0.2230 | 16.68% | MAPE 最优 |
| S1200 | 812 | 0.9247 | 0.2614 | 19.14% | |
| S1500 | 574 | **0.9519** | 0.2359 | 24.95% | **R² 最优** |

---

## 17 维输入特征

`speed, speed_diff, mileage_diff, speed_window20_mean/std, speed_diff_window20_mean, temperature_c/relative_humidity/visibility_km/wind_speed_ms_window20_mean, Low/Mid/High (驾驶模式), cruising_ratio, total_volt, total_current, power`

> power 和 total_current 特征贡献度 >76%

---

## 链式预测

在完整行程上滑动窗口逐段预测 ΔSoC 并累加，重建 SoC 变化曲线。

- **最优策略**: stride=150 (全窗口无重叠)，每 25 分钟更新一次
- **单 trip**: Trip #1159 (4h, ΔSoC=7.99%), 相对误差 3.6%
- **多段拼接**: 7 trips × 12.9h (SoC 100%→36%), 全局相对误差 **0.58%**

---

## 环境说明

| 项目 | 系统默认 | torch_gpu (conda) |
|:-----|:---------|:------------------|
| Python | 3.8.10 | 3.10.20 |
| PyTorch | 2.4.1+cu121 | **2.11.0+cu130** |
| GPU | 不可用 | RTX 5090 可用 |

> 运行 GPU 脚本前必须 `conda activate torch_gpu`

---

## 已知问题

1. **GPU 环境**: 系统默认 Python 3.8 不兼容 RTX 5090，需使用 conda `torch_gpu` 环境
2. **链式预测仅 S1500**: S800 对比待做
3. **单 trip 时长有限**: 最长连续行程约 4.6h，已通过多段拼接方案解决
