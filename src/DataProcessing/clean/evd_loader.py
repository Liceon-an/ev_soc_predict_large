# src/DataProcessing/evd_loader.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import os
from configs.path_config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def convert_units(df):
    """
    对 EVD 气象数据进行单位转换：
    - 温度: °F -> °C
    - 风速: mph -> m/s
    - 能见度: statute miles -> kilometers
    """
    df = df.copy()

    # 1. 温度转换: (°F - 32) * 5/9
    if "HourlyDryBulbTemperature" in df.columns:
        df["HourlyDryBulbTemperature"] = (
            pd.to_numeric(df["HourlyDryBulbTemperature"], errors="coerce") - 32
        ) * 5 / 9
        # 可选：四舍五入到小数点后1位
        df["HourlyDryBulbTemperature"] = df["HourlyDryBulbTemperature"].round(1)

    # 2. 风速转换: mph -> m/s (1 mph ≈ 0.44704 m/s)
    if "HourlyWindSpeed" in df.columns:
        df["HourlyWindSpeed"] = (
            pd.to_numeric(df["HourlyWindSpeed"], errors="coerce") * 0.44704
        ).round(2)

    # 3. 能见度转换: statute miles -> km (1 mile = 1.60934 km)
    if "HourlyVisibility" in df.columns:
        df["HourlyVisibility"] = (
            pd.to_numeric(df["HourlyVisibility"], errors="coerce") * 1.60934
        ).round(2)

    return df


def main():
    # 配置文件路径
    raw_filename = "evd.csv"
    processed_filename = "cleaned_evd.csv"
    input_path = os.path.join(DATA_RAW_DIR, raw_filename)
    output_path = os.path.join(DATA_PROCESSED_DIR, processed_filename)

    # 指定需要保留的列（用于 SOC 预测）
    keep_columns = [
        "DATE",
        "HourlyDryBulbTemperature",   # 将转为 °C
        "HourlyRelativeHumidity",     # 单位 %，无需转换
        "HourlyVisibility",           # 将转为 km
        "HourlyWindSpeed",            # 将转为 m/s
    ]

    # 1. 加载原始数据
    print(f"正在加载原始数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")

    # 2. 仅保留存在的目标列
    existing_cols = [col for col in keep_columns if col in df.columns]
    df = df[existing_cols]
    print(f"保留列后形状: {df.shape}")

    # 3. 标准化时间戳
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        # 可选：删除无法解析的时间行
        initial_len = len(df)
        df = df.dropna(subset=["DATE"])
        if len(df) < initial_len:
            print(f"警告: 删除了 {initial_len - len(df)} 行无效时间戳")
    else:
        raise ValueError("数据中缺少 'DATE' 列，无法对齐时间！")

    # 4. 执行单位转换
    print("正在进行单位转换...")
    df = convert_units(df)

    # 5. （可选）重命名列以提高可读性
    rename_map = {
        "HourlyDryBulbTemperature": "temperature_c",      # 干球温度 (°C)
        "HourlyRelativeHumidity": "relative_humidity",   # 相对湿度 (%)
        "HourlyVisibility": "visibility_km",             # 能见度 (km)
        "HourlyWindSpeed": "wind_speed_ms",              # 风速 (m/s)
    }
    df = df.rename(columns=rename_map)

    # 6. 保存处理后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 保存处理后数据至: {output_path}")
    print(f"最终数据形状: {df.shape}")
    print("列名:", list(df.columns))


if __name__ == "__main__":
    main()