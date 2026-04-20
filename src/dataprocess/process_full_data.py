"""
处理完整数据集的临时脚本
生成最终的时间片段文件：data/split/origin_400s.npz
"""

import sys
import os
from pathlib import Path
import time
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from configs.config_400s import config
from src.dataprocess.time_segment import TimeSegmentDivider

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('temp_full_data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_full_dataset():
    """处理完整数据集"""
    print("=" * 80)
    print("开始处理完整数据集")
    print("=" * 80)
    print(f"输入文件: {config.INPUT_DATA_PATH}")
    print(f"输出文件: {config.OUTPUT_FILE}")
    print(f"配置参数:")
    print(f"  核心步数: {config.CORE_STEPS}步 ({config.CORE_SECONDS}秒)")
    print(f"  总步数: {config.TOTAL_STEPS}步")
    print(f"  步长: {config.STEP_SIZE}步")
    print(f"  断点阈值: {config.BREAK_THRESHOLD}秒")
    print(f"  片段最大时长: {config.MAX_SEGMENT_DURATION}秒")
    print("=" * 80)

    start_time = time.time()

    try:
        # 创建划分器
        divider = TimeSegmentDivider(config)

        # 运行完整流程
        stats = divider.process()

        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 80)
        print("完整数据处理完成!")
        print("=" * 80)
        print(f"处理时间: {processing_time:.2f}秒 ({processing_time/60:.2f}分钟)")
        print(f"总数据点: {stats['total_data_points']:,}")
        print(f"连续段数量: {stats['continuous_segments']}")
        print(f"时间片段数量: {stats['time_segments']:,}")
        if 'coverage_ratio' in stats:
            print(f"数据覆盖率: {stats['coverage_ratio']:.2%}")
            print(f"覆盖数据点: {stats['covered_points']:,}")
        print(f"输出文件: {stats['output_file']}")
        if stats['segment_indices_shape']:
            print(f"片段索引形状: {stats['segment_indices_shape']}")
        print("=" * 80)

        # 验证输出文件
        if Path(config.OUTPUT_FILE).exists():
            import numpy as np
            npz_data = np.load(config.OUTPUT_FILE, allow_pickle=True)
            print(f"\n输出文件验证:")
            print(f"  文件大小: {Path(config.OUTPUT_FILE).stat().st_size / (1024*1024):.2f} MB")
            print(f"  包含数组: {list(npz_data.keys())}")
            print(f"  片段索引形状: {npz_data['segment_indices'].shape}")
            print(f"  元数据键: {list(npz_data['metadata'].item().keys())}")

            # 显示片段统计
            metadata = npz_data['metadata'].item()
            print(f"\n片段统计:")
            print(f"  总片段数: {metadata['total_segments']}")
            print(f"  配置信息:")
            for key, value in metadata['config'].items():
                print(f"    {key}: {value}")

            return True
        else:
            print(f"错误: 输出文件不存在: {config.OUTPUT_FILE}")
            return False

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_data_characteristics():
    """分析数据特征"""
    print("\n" + "=" * 80)
    print("数据特征分析")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np

        # 加载数据基本信息
        data_path = config.INPUT_DATA_PATH
        print(f"分析数据文件: {data_path}")

        # 使用pandas读取前几行获取信息
        df_sample = pd.read_csv(data_path, nrows=1000)
        print(f"数据形状: {df_sample.shape}")
        print(f"列名: {df_sample.columns.tolist()}")

        # 检查时间列
        if config.TIME_COLUMN in df_sample.columns:
            df_sample[config.TIME_COLUMN] = pd.to_datetime(df_sample[config.TIME_COLUMN])
            time_diffs = np.diff(df_sample[config.TIME_COLUMN].values).astype('timedelta64[s]').astype(float)

            print(f"\n时间特征 (基于前1000行):")
            print(f"  时间范围: {df_sample[config.TIME_COLUMN].min()} 到 {df_sample[config.TIME_COLUMN].max()}")
            print(f"  平均时间间隔: {np.mean(time_diffs):.1f}秒")
            print(f"  最小时间间隔: {np.min(time_diffs):.1f}秒")
            print(f"  最大时间间隔: {np.max(time_diffs):.1f}秒")
            print(f"  间隔>500秒的数量: {np.sum(time_diffs > 500)}")

        # 检查速度数据（60.1%为0）
        if 'speed' in df_sample.columns:
            zero_speed_ratio = (df_sample['speed'] == 0).mean()
            print(f"\n速度特征:")
            print(f"  零速度比例: {zero_speed_ratio:.1%}")
            print(f"  平均速度: {df_sample['speed'].mean():.2f} km/h")
            print(f"  最大速度: {df_sample['speed'].max():.2f} km/h")

        return True

    except Exception as e:
        print(f"数据分析失败: {e}")
        return False


if __name__ == "__main__":
    print("完整数据集处理脚本")
    print("=" * 80)

    # 分析数据特征
    analyze_data_characteristics()

    # 处理完整数据集
    success = process_full_dataset()

    if success:
        print("\n" + "=" * 80)
        print("处理成功完成!")
        print("=" * 80)
        print(f"输出文件已生成: {config.OUTPUT_FILE}")
        print(f"日志文件: temp_full_data_processing.log")
    else:
        print("\n" + "=" * 80)
        print("处理失败!")
        print("=" * 80)
        sys.exit(1)