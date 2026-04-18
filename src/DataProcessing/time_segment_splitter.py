#!/usr/bin/env python3
"""
时间片段划分器
将连续时间序列划分为带上下文的片段
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TimeSegmentSplitter:
    """
    时间片段划分器类
    负责将连续时间序列划分为带上下文的片段
    """

    def __init__(self, config: Dict):
        """
        初始化时间片段划分器

        Args:
            config: 配置字典
        """
        self.config = config

        # 从配置中提取参数
        dp_config = config.get('data_processing', {})
        self.sequence_length = dp_config.get('sequence_length', 40)  # 核心序列长度
        self.stride = dp_config.get('stride', 20)  # 滑动步长

        # 新架构特有配置
        enhanced_config = config.get('enhanced_processing', {})
        self.context_size = enhanced_config.get('context_size', 2)  # 上下文大小
        self.min_segment_length = enhanced_config.get('min_segment_length',
                                                     self.sequence_length + 2 * self.context_size)

        # 总序列长度（核心 + 上下文）
        self.total_length = self.sequence_length + 2 * self.context_size

        logger.info(f"时间片段划分器初始化完成")
        logger.info(f"核心序列长度: {self.sequence_length}")
        logger.info(f"上下文大小: {self.context_size}")
        logger.info(f"总序列长度: {self.total_length}")
        logger.info(f"滑动步长: {self.stride}")

    def split_continuous_segment(self, segment_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        将连续时间段划分为带上下文的片段

        Args:
            segment_df: 连续时间段的数据框

        Returns:
            划分后的片段列表
        """
        if len(segment_df) < self.total_length:
            logger.debug(f"连续段长度({len(segment_df)})小于总序列长度({self.total_length})，跳过")
            return []

        segments = []
        n_samples = len(segment_df)

        # 滑动窗口划分
        for start_idx in range(0, n_samples - self.total_length + 1, self.stride):
            end_idx = start_idx + self.total_length

            # 提取片段
            segment = segment_df.iloc[start_idx:end_idx].copy()

            # 标记上下文和核心区域
            segment['is_context'] = False
            segment.iloc[:self.context_size, segment.columns.get_loc('is_context')] = True
            segment.iloc[-self.context_size:, segment.columns.get_loc('is_context')] = True

            # 标记核心序列
            segment['is_core'] = False
            segment.iloc[self.context_size:self.context_size + self.sequence_length,
                        segment.columns.get_loc('is_core')] = True

            # 添加片段元数据
            segment['segment_id'] = len(segments)
            segment['start_idx'] = start_idx
            segment['end_idx'] = end_idx - 1
            segment['context_start'] = start_idx
            segment['context_end'] = end_idx - 1
            segment['core_start'] = start_idx + self.context_size
            segment['core_end'] = start_idx + self.context_size + self.sequence_length - 1

            segments.append(segment)

        logger.debug(f"从连续段中划分出 {len(segments)} 个片段")
        return segments

    def split_multiple_segments(self, segment_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        将多个连续时间段划分为片段

        Args:
            segment_dfs: 连续时间段数据框列表

        Returns:
            所有划分后的片段列表
        """
        all_segments = []

        for i, segment_df in enumerate(segment_dfs):
            logger.debug(f"处理第 {i+1}/{len(segment_dfs)} 个连续段，长度: {len(segment_df)}")

            segments = self.split_continuous_segment(segment_df)
            all_segments.extend(segments)

            logger.debug(f"  生成 {len(segments)} 个片段")

        logger.info(f"总共生成 {len(all_segments)} 个片段")
        return all_segments

    def get_segment_statistics(self, segments: List[pd.DataFrame]) -> Dict:
        """
        获取片段统计信息

        Args:
            segments: 片段列表

        Returns:
            统计信息字典
        """
        if not segments:
            return {}

        stats = {
            'total_segments': len(segments),
            'segment_length': self.total_length,
            'core_length': self.sequence_length,
            'context_size': self.context_size,
            'stride': self.stride
        }

        # 计算每个片段的长度
        segment_lengths = [len(seg) for seg in segments]
        stats['min_segment_length'] = min(segment_lengths)
        stats['max_segment_length'] = max(segment_lengths)
        stats['avg_segment_length'] = np.mean(segment_lengths)

        # 检查是否有时间不连续
        time_gaps = []
        for seg in segments:
            if 'time_diff' in seg.columns:
                time_gaps.extend(seg['time_diff'].tolist())

        if time_gaps:
            stats['max_time_gap'] = max(time_gaps)
            stats['avg_time_gap'] = np.mean(time_gaps)

        return stats

    def validate_segments(self, segments: List[pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        验证片段的有效性

        Args:
            segments: 片段列表

        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []

        for i, seg in enumerate(segments):
            # 检查长度
            if len(seg) != self.total_length:
                errors.append(f"片段 {i}: 长度 {len(seg)} 不等于预期长度 {self.total_length}")

            # 检查时间连续性
            if 'time_diff' in seg.columns:
                max_gap = seg['time_diff'].max()
                if max_gap > self.config.get('data_processing', {}).get('max_time_gap', 500):
                    errors.append(f"片段 {i}: 最大时间间隔 {max_gap} 超过阈值")

        return len(errors) == 0, errors

if __name__ == "__main__":
    # 测试代码
    import yaml

    # 创建测试配置
    test_config = {
        'data_processing': {
            'sequence_length': 40,
            'stride': 20,
            'max_time_gap': 500
        },
        'enhanced_processing': {
            'context_size': 2,
            'min_segment_length': 44
        }
    }

    # 创建划分器
    splitter = TimeSegmentSplitter(test_config)

    print("时间片段划分器测试")
    print("=" * 60)
    print(f"核心序列长度: {splitter.sequence_length}")
    print(f"上下文大小: {splitter.context_size}")
    print(f"总序列长度: {splitter.total_length}")
    print(f"滑动步长: {splitter.stride}")
    print("=" * 60)