"""
时间片段划分算法
用于将连续时间序列数据划分为固定长度的时间片段
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime
import warnings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSegmentDivider:
    """时间片段划分器"""

    def __init__(self, config):
        """
        初始化时间片段划分器

        Args:
            config: 配置对象，包含时间片段划分参数
        """
        self.config = config
        self.data = None
        self.time_series = None
        self.continuous_segments = []
        self.time_segments = []

        logger.info(f"初始化时间片段划分器")
        logger.info(f"配置: 核心{self.config.CORE_STEPS}步, 总{self.config.TOTAL_STEPS}步")
        logger.info(f"时间参数: 断点阈值{self.config.BREAK_THRESHOLD}秒, 最大时长{self.config.MAX_SEGMENT_DURATION}秒")

    def load_data(self, data_path: Path = None) -> pd.DataFrame:
        """
        加载数据

        Args:
            data_path: 数据文件路径，如果为None则使用配置中的路径

        Returns:
            pandas DataFrame
        """
        if data_path is None:
            data_path = self.config.INPUT_DATA_PATH

        logger.info(f"加载数据: {data_path}")

        try:
            # 加载数据
            self.data = pd.read_csv(data_path)

            # 验证数据列
            missing_cols = [col for col in self.config.REQUIRED_COLUMNS if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"数据缺少必要列: {missing_cols}")

            # 转换时间列
            self.data[self.config.TIME_COLUMN] = pd.to_datetime(self.data[self.config.TIME_COLUMN])

            # 按时间排序
            self.data = self.data.sort_values(self.config.TIME_COLUMN).reset_index(drop=True)

            # 提取时间序列
            self.time_series = self.data[self.config.TIME_COLUMN].values

            logger.info(f"数据加载成功: {len(self.data)}行, {len(self.data.columns)}列")
            logger.info(f"时间范围: {self.time_series[0]} 到 {self.time_series[-1]}")

            return self.data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def detect_continuous_segments(self) -> List[Tuple[int, int]]:
        """
        检测连续时间段

        Returns:
            连续段的起始和结束索引列表 [(start_idx, end_idx), ...]
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        logger.info("开始检测连续时间段...")

        # 计算时间间隔（秒）
        time_diffs = np.diff(self.time_series).astype('timedelta64[s]').astype(float)

        # 找到断点（时间间隔 > 断点阈值）
        break_points = np.where(time_diffs > self.config.BREAK_THRESHOLD)[0]

        # 构建连续段
        segments = []
        start_idx = 0

        for break_idx in break_points:
            end_idx = break_idx  # 断点前的索引
            if end_idx - start_idx >= self.config.TOTAL_STEPS:  # 至少能形成一个片段
                segments.append((start_idx, end_idx))
                logger.debug(f"连续段 {len(segments)}: 索引[{start_idx}, {end_idx}], 长度{end_idx-start_idx+1}")
            start_idx = break_idx + 1

        # 添加最后一个段
        if len(self.data) - start_idx >= self.config.TOTAL_STEPS:
            segments.append((start_idx, len(self.data) - 1))
            logger.debug(f"连续段 {len(segments)}: 索引[{start_idx}, {len(self.data)-1}], 长度{len(self.data)-start_idx}")

        self.continuous_segments = segments

        logger.info(f"检测到 {len(segments)} 个连续时间段")
        for i, (start, end) in enumerate(segments):
            segment_len = end - start + 1
            # 修复时间差计算：numpy timedelta64转换为秒
            time_diff = self.time_series[end] - self.time_series[start]
            time_span = time_diff / np.timedelta64(1, 's')
            logger.info(f"  段{i+1}: 索引[{start}, {end}], 长度{segment_len}, 时间跨度{time_span:.1f}秒")

        return segments

    def generate_time_segments(self) -> List[Dict[str, Any]]:
        """
        生成时间片段

        Returns:
            时间片段信息列表，每个片段包含索引和元数据
        """
        if not self.continuous_segments:
            self.detect_continuous_segments()

        logger.info("开始生成时间片段...")

        segments = []
        segment_id = 0

        for seg_idx, (start_idx, end_idx) in enumerate(self.continuous_segments):
            segment_data_len = end_idx - start_idx + 1

            # 计算该段能生成的片段数量
            max_segments = (segment_data_len - self.config.TOTAL_STEPS) // self.config.STEP_SIZE + 1

            logger.info(f"连续段 {seg_idx+1}: 数据长度{segment_data_len}, 最多可生成{max_segments}个片段")

            for i in range(max_segments):
                segment_start = start_idx + i * self.config.STEP_SIZE
                segment_end = segment_start + self.config.TOTAL_STEPS - 1

                # 检查索引边界
                if segment_end > end_idx:
                    break

                # 检查时间连续性
                time_start = self.time_series[segment_start]
                time_end = self.time_series[segment_end]
                # 修复时间差计算：numpy timedelta64转换为秒
                time_diff = time_end - time_start
                segment_duration = time_diff / np.timedelta64(1, 's')

                if segment_duration > self.config.MAX_SEGMENT_DURATION:
                    if self.config.VERBOSE:
                        logger.warning(f"片段 {segment_id} 时长{segment_duration:.1f}秒超过限制{self.config.MAX_SEGMENT_DURATION}秒，跳过")
                    continue

                # 创建片段信息
                segment_info = {
                    'id': segment_id,
                    'continuous_segment_idx': seg_idx,
                    'start_idx': segment_start,
                    'end_idx': segment_end,
                    'indices': list(range(segment_start, segment_end + 1)),
                    'start_time': time_start,
                    'end_time': time_end,
                    'duration_seconds': segment_duration,
                    'core_start_idx': segment_start + self.config.CONTEXT_STEPS,
                    'core_end_idx': segment_end - self.config.CONTEXT_STEPS
                }

                segments.append(segment_info)
                segment_id += 1

                if self.config.VERBOSE and segment_id % 100 == 0:
                    logger.debug(f"已生成 {segment_id} 个片段")

        self.time_segments = segments

        logger.info(f"成功生成 {len(segments)} 个时间片段")

        # 统计信息
        if segments:
            durations = [s['duration_seconds'] for s in segments]
            logger.info(f"片段时长统计: 平均{np.mean(durations):.1f}秒, 最小{np.min(durations):.1f}秒, 最大{np.max(durations):.1f}秒")

        return segments

    def save_segments(self, output_path: Path = None) -> Path:
        """
        保存时间片段到NPZ文件

        Args:
            output_path: 输出文件路径，如果为None则使用配置中的路径

        Returns:
            保存的文件路径
        """
        if not self.time_segments:
            self.generate_time_segments()

        if output_path is None:
            output_path = self.config.OUTPUT_FILE

        logger.info(f"保存时间片段到: {output_path}")

        try:
            # 准备保存的数据
            segment_indices = np.array([s['indices'] for s in self.time_segments], dtype=np.int32)

            # 保存元数据
            metadata = {
                'config': {
                    'core_steps': self.config.CORE_STEPS,
                    'context_steps': self.config.CONTEXT_STEPS,
                    'total_steps': self.config.TOTAL_STEPS,
                    'step_size': self.config.STEP_SIZE,
                    'break_threshold': self.config.BREAK_THRESHOLD,
                    'max_segment_duration': self.config.MAX_SEGMENT_DURATION,
                    'time_resolution': self.config.TIME_RESOLUTION
                },
                'segment_info': [
                    {
                        'id': s['id'],
                        'continuous_segment_idx': s['continuous_segment_idx'],
                        'start_idx': s['start_idx'],
                        'end_idx': s['end_idx'],
                        'start_time': str(s['start_time']),  # 转换为字符串
                        'end_time': str(s['end_time']),      # 转换为字符串
                        'duration_seconds': s['duration_seconds'],
                        'core_start_idx': s['core_start_idx'],
                        'core_end_idx': s['core_end_idx']
                    }
                    for s in self.time_segments
                ],
                'data_shape': self.data.shape if self.data is not None else None,
                'total_segments': len(self.time_segments)
            }

            # 保存到NPZ文件
            np.savez_compressed(
                output_path,
                segment_indices=segment_indices,
                metadata=metadata
            )

            logger.info(f"保存成功: {output_path}")
            logger.info(f"  片段数量: {len(self.time_segments)}")
            logger.info(f"  片段索引形状: {segment_indices.shape}")

            return output_path

        except Exception as e:
            logger.error(f"保存失败: {e}")
            raise

    def process(self, data_path: Path = None, output_path: Path = None) -> Dict[str, Any]:
        """
        完整处理流程

        Args:
            data_path: 输入数据路径
            output_path: 输出文件路径

        Returns:
            处理结果统计信息
        """
        logger.info("=" * 60)
        logger.info("开始时间片段划分处理")
        logger.info("=" * 60)

        # 加载数据
        self.load_data(data_path)

        # 检测连续段
        self.detect_continuous_segments()

        # 生成时间片段
        self.generate_time_segments()

        # 保存结果
        saved_path = self.save_segments(output_path)

        # 统计信息
        stats = {
            'total_data_points': len(self.data),
            'continuous_segments': len(self.continuous_segments),
            'time_segments': len(self.time_segments),
            'output_file': str(saved_path),
            'segment_indices_shape': None
        }

        if self.time_segments:
            stats['segment_indices_shape'] = (len(self.time_segments), self.config.TOTAL_STEPS)

            # 计算覆盖率
            covered_indices = set()
            for segment in self.time_segments:
                covered_indices.update(segment['indices'])

            stats['coverage_ratio'] = len(covered_indices) / len(self.data)
            stats['covered_points'] = len(covered_indices)

        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"  总数据点: {stats['total_data_points']}")
        logger.info(f"  连续段数量: {stats['continuous_segments']}")
        logger.info(f"  时间片段数量: {stats['time_segments']}")
        if 'coverage_ratio' in stats:
            logger.info(f"  数据覆盖率: {stats['coverage_ratio']:.2%}")
        logger.info(f"  输出文件: {stats['output_file']}")
        logger.info("=" * 60)

        return stats


def main():
    import sys
    import argparse
    sys.path.append(".")

    parser = argparse.ArgumentParser(description="时间片段划分")
    parser.add_argument("--window", type=str, default="400",
                        help="窗口大小(秒)，如 200/400/600/800 (默认: 400)")
    args = parser.parse_args()

    try:
        cfg_name = "config_{}s".format(args.window)
        cfg_module = __import__("configs." + cfg_name, fromlist=["config"])
        config = cfg_module.config
    except ImportError:
        try:
            from configs.config_400s import config
            print("未找到 {}，使用默认 400s".format(cfg_name))
        except ImportError:
            print("无法加载配置: configs.{}".format(cfg_name))
            sys.exit(1)

    try:
        divider = TimeSegmentDivider(config)
        stats = divider.process()

        print("\n处理统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"处理失败: {e}")
        raise


if __name__ == "__main__":
    main()