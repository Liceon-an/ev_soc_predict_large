#!/usr/bin/env python3
"""
build_dataset.py - 从时间片段构建训练/验证/测试数据集

从 feature_data.csv + origin_{N}s.npz 提取片段特征、计算标签，
随机打乱后按比例划分 train/val/test，Z-Score 标准化后保存。

用法:
  python src/dataprocess/build_dataset.py

输出 (根据 config.CORE_SECONDS 自动变更路径):
  data/train/split_{CORE_SECONDS}s/
    train_data.npz
    val_data.npz
    test_data.npz
    scaler.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 用户可调参数
# ============================================================

FEATURE_COLUMNS = [
    "speed", "speed_diff", "mileage_diff",
    "speed_window20_mean", "speed_diff_window20_mean",
    "temperature_c_window20_mean", "relative_humidity_window20_mean",
    "visibility_km_window20_mean", "wind_speed_ms_window20_mean",
    "speed_window20_std",
    "Low", "Mid", "High", "cruising_ratio"
]

N_DROP_FIRST = 2
N_DROP_LAST = 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42


def load_data(config):
    logger.info("加载特征数据: %s", config.INPUT_DATA_PATH)
    df = pd.read_csv(config.INPUT_DATA_PATH)
    logger.info("  形状: %s", df.shape)

    npz_path = config.OUTPUT_FILE
    logger.info("加载片段索引: %s", npz_path)
    meta = np.load(npz_path, allow_pickle=True)
    segment_indices = meta["segment_indices"]
    logger.info("  形状: %s", segment_indices.shape)
    return df, segment_indices


def filter_segments(segment_indices):
    n = len(segment_indices)
    start, end = N_DROP_FIRST, n - N_DROP_LAST
    logger.info("丢弃前%d+后%d片段: %d -> %d", N_DROP_FIRST, N_DROP_LAST, n, end - start)
    return segment_indices[start:end]


def extract_features_and_labels(df, segment_indices):
    n_seg = len(segment_indices)
    seq_len = segment_indices.shape[1]
    n_feat = len(FEATURE_COLUMNS)

    logger.info("提取特征: %d片段 x %d步 x %d特征", n_seg, seq_len, n_feat)

    X = np.zeros((n_seg, seq_len, n_feat), dtype=np.float32)
    y_main = np.zeros(n_seg, dtype=np.float32)
    y_aux = np.zeros(n_seg, dtype=np.float32)
    nan_segments = []

    for i in range(n_seg):
        rows = segment_indices[i]
        seg_data = df.iloc[rows]

        X[i] = seg_data[FEATURE_COLUMNS].values.astype(np.float32)
        y_main[i] = seg_data["power"].sum()
        y_aux[i] = seg_data["refined_soc"].iloc[0] - seg_data["refined_soc"].iloc[-1]

        if np.any(np.isnan(X[i])):
            nan_segments.append(i)

    if nan_segments:
        logger.warning("发现 %d 个片段含 NaN: %s", len(nan_segments), nan_segments[:10])
    else:
        logger.info("所有片段特征无 NaN")

    logger.info("total_energy: mean=%.2f, std=%.2f, [%.2f, %.2f]",
                y_main.mean(), y_main.std(), y_main.min(), y_main.max())
    logger.info("delta_soc:     mean=%.4f, std=%.4f, [%.4f, %.4f]",
                y_aux.mean(), y_aux.std(), y_aux.min(), y_aux.max())

    return X, y_main, y_aux


def shuffle_data(X, y_main, y_aux):
    n = len(X)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    X_s = X[idx]
    y_main_s = y_main[idx]
    y_aux_s = y_aux[idx]
    logger.info("随机打散: %d 个片段 (seed=%d)", n, RANDOM_SEED)
    return X_s, y_main_s, y_aux_s


def split_data(X, y_main, y_aux):
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {}
    for name, (s, e) in [
        ("train", (0, train_end)),
        ("val", (train_end, val_end)),
        ("test", (val_end, n)),
    ]:
        splits[name] = {
            "X": X[s:e],
            "y_main": y_main[s:e],
            "y_aux": y_aux[s:e],
        }
        logger.info("%s: X=%s", name, str(splits[name]["X"].shape))
    return splits


def compute_scaler(X_train):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    for i, col in enumerate(FEATURE_COLUMNS):
        logger.info("  %s: mean=%.4f, std=%.4f", col, mean[i], std[i])
    return mean, std


def apply_scaler(splits, mean, std):
    for name in splits:
        splits[name]["X"] = (splits[name]["X"] - mean) / std
    logger.info("Z-Score 标准化完成")


def save_dataset(splits, mean, std, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train", "val", "test"]:
        d = splits[name]
        path = output_dir / ("%s_data.npz" % name)
        np.savez_compressed(path, X=d["X"], y_main=d["y_main"], y_aux=d["y_aux"])
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info("保存 %s: %s (%.2f MB)", name, str(path), size_mb)

    scaler_path = output_dir / "scaler.npz"
    np.savez_compressed(
        scaler_path,
        mean=mean,
        std=std,
        feature_cols=np.array(FEATURE_COLUMNS, dtype=object),
    )
    logger.info("保存 scaler: %s", str(scaler_path))
    return output_dir


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config_400s import config

    logger.info("=" * 60)
    logger.info("构建数据集 | %ds 核心 | %d 步/片段", config.CORE_SECONDS, config.CORE_STEPS)
    logger.info("=" * 60)

    df, seg_idx = load_data(config)
    seg_idx = filter_segments(seg_idx)
    X, y_main, y_aux = extract_features_and_labels(df, seg_idx)
    X, y_main, y_aux = shuffle_data(X, y_main, y_aux)
    splits = split_data(X, y_main, y_aux)
    mean, std = compute_scaler(splits["train"]["X"])
    apply_scaler(splits, mean, std)

    output_dir = Path("data/train") / ("split_%ds" % config.CORE_SECONDS)
    save_dataset(splits, mean, std, output_dir)

    logger.info("=" * 60)
    logger.info("数据集构建完成")
    logger.info("输出目录: %s", str(output_dir.resolve()))
    logger.info("特征: %d 个", len(FEATURE_COLUMNS))
    for name in ["train", "val", "test"]:
        d = splits[name]
        logger.info("  %5s: X=%s  y_main=[%8.2f, %8.2f]  y_aux=[%8.4f, %8.4f]",
                    name,
                    str(d["X"].shape),
                    d["y_main"].min(),
                    d["y_main"].max(),
                    d["y_aux"].min(),
                    d["y_aux"].max())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
