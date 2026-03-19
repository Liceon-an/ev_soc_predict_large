# src/DataProcessing/dataset.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import os
from configs.path_config import DATA_ALIGNED_DIR, DATA_PROCESSED_DIR


def split_into_continuous_trips(df, max_gap_seconds=30):
    """
    将数据按连续驾驶段分割（时间间隔 > max_gap_seconds 视为新行程）
    返回: list of DataFrames, 每个 DataFrame 是一个连续行程
    """
    df = df.sort_values("DATE").reset_index(drop=True)
    df["time_diff"] = df["DATE"].diff().dt.total_seconds().fillna(0)
    
    # 标记新行程起点（时间差 > 阈值 或 第一行）
    df["is_new_trip"] = (df["time_diff"] > max_gap_seconds) | (df.index == 0)
    df["trip_id"] = df["is_new_trip"].cumsum()
    
    trips = []
    for trip_id, group in df.groupby("trip_id"):
        if len(group) >= 2:  # 至少2条记录才视为有效行程
            trips.append(group.reset_index(drop=True))
    
    print(f"检测到 {len(trips)} 个连续驾驶段")
    return trips


def extract_segment_features(segment_df):
    """从单个连续片段中提取统计特征和标签"""
    features = {}
    
    # 车辆动态特征
    features["mean_speed"] = segment_df["speed"].mean()
    features["std_speed"] = segment_df["speed"].std()
    features["max_speed"] = segment_df["speed"].max()
    features["mean_current"] = segment_df["total_current"].mean()
    features["mean_volt"] = segment_df["total_volt"].mean()
    features["mean_power"] = (segment_df["total_volt"] * segment_df["total_current"]).mean()
    
    # 电池温度
    features["mean_max_temp"] = segment_df["max_temp"].mean()
    features["mean_min_temp"] = segment_df["min_temp"].mean()
    features["mean_temp_diff"] = (segment_df["max_temp"] - segment_df["min_temp"]).mean()
    
    # 环境特征
    features["mean_temperature_c"] = segment_df["temperature_c"].mean()
    features["mean_humidity"] = segment_df["relative_humidity"].mean()
    features["mean_wind"] = segment_df["wind_speed_ms"].mean()
    features["mean_visibility"] = segment_df["visibility_km"].mean()
    
    # 标签：SOC 消耗（正数表示消耗）
    soc_start = segment_df["standard_soc"].iloc[0]
    soc_end = segment_df["standard_soc"].iloc[-1]
    features["delta_soc"] = soc_start - soc_end
    
    return features


def build_segment_dataset(input_path, output_path, window_size=40, step=20, min_trip_length=50):
    """
    从对齐数据构建片段级数据集
    :param window_size: 片段长度（40 = 400s）
    :param step: 滑动步长
    :param min_trip_length: 行程最小长度（避免太短的行程）
    """
    # 1. 加载数据
    print("正在加载对齐数据...")
    df = pd.read_csv(input_path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    # 2. 按连续行程分割
    trips = split_into_continuous_trips(df, max_gap_seconds=30)
    
    # 3. 在每个行程内切分片段
    all_segments = []
    total_original_rows = 0
    
    for trip in trips:
        total_original_rows += len(trip)
        if len(trip) < min_trip_length:
            continue  # 跳过太短的行程
        
        # 在当前行程内滑动窗口
        for i in range(0, len(trip) - window_size + 1, step):
            segment = trip.iloc[i:i + window_size]
            
            # 跳过包含 NaN 的片段
            if segment[["standard_soc", "speed", "total_current"]].isna().any().any():
                continue
            
            # 提取特征
            feat = extract_segment_features(segment)
            all_segments.append(feat)
    
    print(f"原始总行数: {total_original_rows}")
    print(f"生成片段样本数: {len(all_segments)}")
    
    # 4. 保存
    if all_segments:
        dataset = pd.DataFrame(all_segments)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print(f"✅ 数据集已保存至: {output_path}")
        print(f"特征维度: {dataset.shape[1] - 1} (含标签)")
        print("列名:", list(dataset.columns))
    else:
        print("❌ 未生成任何样本，请检查数据或参数！")


if __name__ == "__main__":
    input_file = os.path.join(DATA_ALIGNED_DIR, "aligned_data.csv")
    output_file = os.path.join(DATA_PROCESSED_DIR, "segment_dataset.csv")
    
    # 参数说明:
    # - window_size=40 → 400s (10s/step)
    # - step=20 → 每200s滑动一次（半重叠）
    # - min_trip_length=50 → 至少500s的行程才处理
    build_segment_dataset(
        input_path=input_file,
        output_path=output_file,
        window_size=40,
        step=20,
        min_trip_length=50
    )