# src/DataProcessing/data_loader.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import os
from configs.path_config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def main():
    # 配置文件路径
    raw_filename = "Dataset1.csv"
    processed_filename = "cleaned_data.csv"
    input_path = os.path.join(DATA_RAW_DIR, raw_filename)
    output_path = os.path.join(DATA_PROCESSED_DIR, processed_filename)

    # 保留对 SOC 预测有用的列
    keep_columns = [
        "yr_modahrmn",
        "charging_status",
        "speed",
        "mileage",
        "total_volt",
        "total_current",
        "max_cell_volt",
        "min_cell_volt",
        "max_temp",
        "min_temp",
        "standard_soc",
    ]

    # 1. 加载数据
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")

    # 2. 只保留存在的指定列
    existing_cols = [col for col in keep_columns if col in df.columns]
    df = df[existing_cols]
    print(f"保留列后形状: {df.shape}")

    # 3. 剔除 charging_status == 1 的行
    if "charging_status" in df.columns:
        before = len(df)
        df = df[df["charging_status"] != 1]
        after = len(df)
        print(f"剔除 charging_status=1 的行: {before} → {after}")
        df = df.drop(columns=["charging_status"])
    else:
        print("'charging_status' 列不存在，跳过行过滤")

    # 4. 处理时间戳：解析、移除时区、保留本地时间
    if "yr_modahrmn" in df.columns:
        df["yr_modahrmn"] = pd.to_datetime(df["yr_modahrmn"], errors="coerce")

        # 如果有时区信息，移除时区但保留本地时刻
        if df["yr_modahrmn"].dt.tz is not None:
            df["yr_modahrmn"] = df["yr_modahrmn"].dt.tz_localize(None)

        # 剔除无效时间戳
        invalid_count = df["yr_modahrmn"].isna().sum()
        if invalid_count > 0:
            print(f"剔除 {invalid_count} 行无效时间戳")
            df = df.dropna(subset=["yr_modahrmn"])

        # 排序
        df = df.sort_values("yr_modahrmn").reset_index(drop=True)

        # ✅ 关键修改：将列名从 'yr_modahrmn' 改为 'DATE'
        df = df.rename(columns={"yr_modahrmn": "DATE"})
    else:
        print("警告: 'yr_modahrmn' 列不存在，无法重命名为 'DATE'")

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"保存至: {output_path}")
    print(f"最终数据形状: {df.shape}")


if __name__ == "__main__":
    main()