# src/DataProcessing/align_vehicle_weather.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import os
from configs.path_config import DATA_PROCESSED_DIR, DATA_ALIGNED_DIR


def main():
    # 文件路径
    vehicle_file = os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")
    weather_file = os.path.join(DATA_PROCESSED_DIR, "cleaned_evd.csv")
    output_file = os.path.join(DATA_ALIGNED_DIR, "aligned_data.csv")

    # 1. 加载数据
    vehicle_df = pd.read_csv(vehicle_file)
    print(f"车辆数据形状: {vehicle_df.shape}")

    weather_df = pd.read_csv(weather_file)
    print(f"天气数据形状: {weather_df.shape}")

    # 2. 确保 DATE 列为 datetime（无时区）
    vehicle_df["DATE"] = pd.to_datetime(vehicle_df["DATE"])
    weather_df["DATE"] = pd.to_datetime(weather_df["DATE"])

    # 3. 按 DATE 排序（merge_asof 要求）
    vehicle_df = vehicle_df.sort_values("DATE").reset_index(drop=True)
    weather_df = weather_df.sort_values("DATE").reset_index(drop=True)

    # 4. 时间对齐：为每条车辆记录匹配最近的天气（不超过30分钟前的天气）
    print("正在执行时间对齐 (merge_asof)...")
    aligned_df = pd.merge_asof(
        vehicle_df,
        weather_df,
        on="DATE",
        direction="backward",      # 使用当前时刻或之前的最近天气
        tolerance=pd.Timedelta("30min")  # 最大容忍30分钟滞后（可调）
    )

    # 5. 检查对齐效果
    matched_count = aligned_df.dropna(subset=["temperature_c"]).shape[0]
    total_count = aligned_df.shape[0]
    print(f"成功匹配天气的记录: {matched_count} / {total_count} ({matched_count/total_count:.1%})")

    # 可选：删除未匹配到天气的行
    # aligned_df = aligned_df.dropna(subset=["temperature_c"])

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    aligned_df.to_csv(output_file, index=False)
    print(f"✅ 对齐完成！保存至: {output_file}")
    print(f"最终数据形状: {aligned_df.shape}")
    print("列名:", list(aligned_df.columns))


if __name__ == "__main__":
    main()