# src/FeatureEngineering/refine_soc_with_coulomb_counting.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import os
from configs.path_config import DATA_ALIGNED_DIR, DATA_PROCESSED_DIR


def split_into_continuous_trips(df, max_gap_seconds=30):
    """按时间间隙分割连续行程"""
    df = df.sort_values("DATE").reset_index(drop=True)
    df["time_diff"] = df["DATE"].diff().dt.total_seconds().fillna(0)
    df["is_new_trip"] = (df["time_diff"] > max_gap_seconds) | (df.index == 0)
    df["trip_id"] = df["is_new_trip"].cumsum()
    trips = []
    for trip_id, group in df.groupby("trip_id"):
        if len(group) >= 10:  # 至少10条记录
            trips.append(group.reset_index(drop=True))
    return trips


def refine_soc_in_trip(trip_df, dt_seconds=10, min_soc_change=0.5):
    """
    对单个连续行程进行 SOC 精细化
    :param trip_df: 单个行程 DataFrame
    :param dt_seconds: 采样间隔（秒）
    :param min_soc_change: 最小 SOC 变化阈值（%），避免噪声段
    :return: 添加 refined_soc 列的 DataFrame
    """
    trip = trip_df.copy()
    
    # 1. 计算每个时间步的放电量（Ah），仅考虑放电（I > 0）
    # 注：若想包含再生制动，可去掉 clip，但需确保 SOC 单调
    trip["current_discharge"] = np.clip(trip["total_current"], 0, None)  # 忽略负电流（充电）
    trip["delta_q_ah"] = trip["current_discharge"] * (dt_seconds / 3600.0)  # Ah
    
    # 2. 获取起止 SOC
    soc_start = trip["standard_soc"].iloc[0]
    soc_end = trip["standard_soc"].iloc[-1]
    delta_soc_obs = soc_start - soc_end  # 应为正数
    
    # 3. 跳过无效行程
    if delta_soc_obs < min_soc_change:
        trip["refined_soc"] = trip["standard_soc"]  # 不处理
        return trip
    
    total_q_ah = trip["delta_q_ah"].sum()
    if total_q_ah <= 0:
        trip["refined_soc"] = trip["standard_soc"]
        return trip
    
    # 4. 计算有效容量 C (Ah)
    C_ah = total_q_ah / (delta_soc_obs / 100.0)  # Ah per 100% SOC
    
    # 5. 重构 SOC：从起点开始积分
    cumulative_q = trip["delta_q_ah"].cumsum()
    trip["refined_soc"] = soc_start - (cumulative_q / C_ah) * 100.0
    
    # 6. 可选：平滑或限制范围 [0, 100]
    trip["refined_soc"] = np.clip(trip["refined_soc"], 0, 100)
    
    print(f"行程 {trip['trip_id'].iloc[0]}: ΔSOC={delta_soc_obs:.2f}%, C={C_ah:.1f} Ah")
    return trip


def refine_soc_dataset(input_path, output_path, dt_seconds=10):
    """
    主函数：对整个数据集进行 SOC 精细化
    """
    print("正在加载对齐数据...")
    df = pd.read_csv(input_path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    # 分割连续行程
    trips = split_into_continuous_trips(df, max_gap_seconds=30)
    print(f"共检测到 {len(trips)} 个连续行程")
    
    refined_trips = []
    for trip in trips:
        refined_trip = refine_soc_in_trip(trip, dt_seconds=dt_seconds, min_soc_change=0.5)
        refined_trips.append(refined_trip)
    
    # 合并结果
    refined_df = pd.concat(refined_trips, ignore_index=True)
    refined_df = refined_df.sort_values("DATE").reset_index(drop=True)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    refined_df.to_csv(output_path, index=False)
    print(f"✅ 精细化 SOC 数据已保存至: {output_path}")
    print("新增列: 'refined_soc'")
    
    # 统计改进效果
    mae_original = np.mean(np.abs(np.diff(df["standard_soc"])))
    mae_refined = np.mean(np.abs(np.diff(refined_df["refined_soc"])))
    print(f"SOC 变化平滑度（平均绝对差）: 原始={mae_original:.4f}, 精细化={mae_refined:.4f}")


if __name__ == "__main__":
    input_file = os.path.join(DATA_ALIGNED_DIR, "aligned_data.csv")
    output_file = os.path.join(DATA_PROCESSED_DIR, "aligned_data_refined_soc.csv")
    
    refine_soc_dataset(
        input_path=input_file,
        output_path=output_file,
        dt_seconds=10
    )