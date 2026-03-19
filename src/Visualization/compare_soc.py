# src/Visualization/compare_soc_interactive.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from configs.path_config import DATA_PROCESSED_DIR


def plot_trip_soc(trip_data, trip_id):
    """绘制单个行程的 SOC 对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(trip_data['DATE'], trip_data['standard_soc'],
             label='SOC Values before Ampere-hour(Ah) Integration Method', linewidth=2, alpha=0.8, marker='o', markersize=3)
    plt.plot(trip_data['DATE'], trip_data['refined_soc'],
             label='SOC Values after Ampere-hour(Ah) Integration Method', linewidth=2, linestyle='--')

    plt.title(f'SOC Comparison', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('SOC (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()


def interactive_soc_viewer(df):
    """交互式查看多个随机行程"""
    # 获取所有有效行程
    trip_lengths = df.groupby('trip_id').size()
    valid_trips = trip_lengths[trip_lengths >= 10].index.tolist()

    if not valid_trips:
        print("❌ 未找到足够长的连续行程")
        return

    print(f"✅ 共有 {len(valid_trips)} 个有效行程可供查看")
    print("按任意键关闭当前图后，将自动显示下一个行程。\n输入 'q' 并回车可退出。")

    while True:
        # 随机选一个行程
        selected_trip = np.random.choice(valid_trips)
        trip_data = df[df['trip_id'] == selected_trip].sort_values('DATE').reset_index(drop=True)

        # 打印信息
        start_time = trip_data['DATE'].iloc[0]
        end_time = trip_data['DATE'].iloc[-1]
        duration_sec = (end_time - start_time).total_seconds()
        delta_soc_orig = trip_data['standard_soc'].iloc[0] - trip_data['standard_soc'].iloc[-1]
        delta_soc_refined = trip_data['refined_soc'].iloc[0] - trip_data['refined_soc'].iloc[-1]

        print("\n" + "="*60)
        print(f"▶ 当前行程 ID: {selected_trip}")
        print(f"  时间: {start_time} → {end_time}")
        print(f"  时长: {duration_sec:.0f} 秒 ({duration_sec/60:.1f} 分钟)")
        print(f"  ΔSOC (原始): {delta_soc_orig:.2f}%")
        print(f"  ΔSOC (精细化): {delta_soc_refined:.2f}%")
        print("="*60)

        # 绘图（会阻塞，直到用户关闭窗口）
        plot_trip_soc(trip_data, selected_trip)

        # 询问是否继续
        user_input = input("按 Enter 查看下一个行程，输入 'q' 退出: ").strip().lower()
        if user_input == 'q':
            print("👋 退出查看器")
            break


def main():
    input_file = os.path.join(DATA_PROCESSED_DIR, "aligned_data_refined_soc.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"请先生成数据: {input_file}")

    df = pd.read_csv(input_file)
    df['DATE'] = pd.to_datetime(df['DATE'])

    if 'trip_id' not in df.columns:
        raise ValueError("数据缺少 'trip_id' 列，请先运行 refine_soc 脚本")

    interactive_soc_viewer(df)


if __name__ == "__main__":
    main()