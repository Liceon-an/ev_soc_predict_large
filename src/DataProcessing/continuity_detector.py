#!/usr/bin/env python3
"""
连续段检测器
识别数据中的连续时间段，避免跨越时间断点
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ContinuityDetector:
    """
    连续段检测器类
    负责识别数据中的连续时间段
    """

    def __init__(self, config: Dict):
        """
        初始化连续段检测器

        Args:
            config: 配置字典
        """
        self.config = config

        # 从配置中提取参数
        dp_config = config.get('data_processing', {})
        self.max_time_gap = dp_config.get('max_time_gap', 500)  # 最大时间间隔阈值（秒）
        self.time_col = dp_config.get('time_col', 'DATE')  # 时间列名

        # 新架构特有配置
        enhanced_config = config.get('enhanced_processing', {})
        self.enabled = enhanced_config.get('continuity_detection', True)

        logger.info(f"连续段检测器初始化完成")
        logger.info(f"最大时间间隔阈值: {self.max_time_gap}秒")
        logger.info(f"时间列: {self.time_col}")
        logger.info(f"启用状态: {self.enabled}")

    def detect_continuous_segments(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        检测数据中的连续时间段

        Args:
            df: 输入数据框，必须包含时间列

        Returns:
            连续时间段的数据框列表
        """
        if not self.enabled:
            logger.info("连续段检测已禁用，返回完整数据作为单个段")
            return [df]

        if self.time_col not in df.columns:
            logger.warning(f"时间列 '{self.time_col}' 不存在，返回完整数据作为单个段")
            return [df]

        # 确保数据按时间排序
        df_sorted = df.sort_values(self.time_col).copy()
        df_sorted = df_sorted.reset_index(drop=True)

        # 计算时间间隔
        times = pd.to_datetime(df_sorted[self.time_col])
        time_diffs = times.diff().dt.total_seconds()

        # 第一行的时间间隔设为0
        time_diffs.iloc[0] = 0
        df_sorted['time_diff'] = time_diffs

        # 识别断点（时间间隔超过阈值的位置）
        breakpoints = np.where(time_diffs > self.max_time_gap)[0]

        # 如果没有断点，返回完整数据
        if len(breakpoints) == 0:
            logger.info("未发现时间断点，数据是连续的")
            return [df_sorted]

        # 根据断点划分连续段
        segments = []
        start_idx = 0

        for break_idx in breakpoints:
            # 提取当前连续段（不包括断点）
            segment = df_sorted.iloc[start_idx:break_idx].copy()
            if len(segment) > 0:
                segments.append(segment)

            # 更新起始索引（断点后的第一行）
            start_idx = break_idx

        # 添加最后一个段
        if start_idx < len(df_sorted):
            segment = df_sorted.iloc[start_idx:].copy()
            if len(segment) > 0:
                segments.append(segment)

        logger.info(f"检测到 {len(breakpoints)} 个断点，划分出 {len(segments)} 个连续段")
        return segments

    def get_segment_statistics(self, segments: List[pd.DataFrame]) -> Dict:
        """
        获取连续段统计信息

        Args:
            segments: 连续段列表

        Returns:
            统计信息字典
        """
        if not segments:
            return {}

        stats = {
            'total_segments': len(segments),
            'total_samples': sum(len(seg) for seg in segments),
            'max_time_gap_threshold': self.max_time_gap
        }

        # 计算每个段的长度
        segment_lengths = [len(seg) for seg in segments]
        stats['min_segment_length'] = min(segment_lengths)
        stats['max_segment_length'] = max(segment_lengths)
        stats['avg_segment_length'] = np.mean(segment_lengths)
        stats['median_segment_length'] = np.median(segment_lengths)

        # 计算每个段的时间跨度
        time_spans = []
        for seg in segments:
            if self.time_col in seg.columns:
                times = pd.to_datetime(seg[self.time_col])
                time_span = (times.max() - times.min()).total_seconds()
                time_spans.append(time_span)

        if time_spans:
            stats['min_time_span'] = min(time_spans)
            stats['max_time_span'] = max(time_spans)
            stats['avg_time_span'] = np.mean(time_spans)

        # 计算每个段的平均时间间隔
        avg_time_gaps = []
        for seg in segments:
            if 'time_diff' in seg.columns:
                avg_gap = seg['time_diff'].mean()
                avg_time_gaps.append(avg_gap)

        if avg_time_gaps:
            stats['min_avg_time_gap'] = min(avg_time_gaps)
            stats['max_avg_time_gap'] = max(avg_time_gaps)
            stats['avg_time_gap'] = np.mean(avg_time_gaps)

        # 识别长段和短段
        long_segments = [seg for seg in segments if len(seg) >= 100]
        short_segments = [seg for seg in segments if len(seg) < 100]

        stats['long_segments_count'] = len(long_segments)
        stats['short_segments_count'] = len(short_segments)
        stats['long_segments_samples'] = sum(len(seg) for seg in long_segments)
        stats['short_segments_samples'] = sum(len(seg) for seg in short_segments)

        return stats

    def analyze_breakpoints(self, df: pd.DataFrame) -> Dict:
        """
        分析数据中的断点

        Args:
            df: 输入数据框

        Returns:
            断点分析信息
        """
        if self.time_col not in df.columns:
            return {'error': f"时间列 '{self.time_col}' 不存在"}

        # 确保数据按时间排序
        df_sorted = df.sort_values(self.time_col).copy()
        times = pd.to_datetime(df_sorted[self.time_col])
        time_diffs = times.diff().dt.total_seconds()
        time_diffs.iloc[0] = 0

        # 识别断点
        break_indices = np.where(time_diffs > self.max_time_gap)[0]
        break_gaps = time_diffs.iloc[break_indices].values

        analysis = {
            'total_breakpoints': len(break_indices),
            'max_gap_threshold': self.max_time_gap,
            'breakpoints': []
        }

        # 收集每个断点的详细信息
        for i, (idx, gap) in enumerate(zip(break_indices, break_gaps)):
            if idx > 0 and idx < len(df_sorted):
                before_time = times.iloc[idx - 1]
                after_time = times.iloc[idx]
                time_between = after_time - before_time

                break_info = {
                    'index': int(idx),
                    'gap_seconds': float(gap),
                    'before_time': before_time.isoformat(),
                    'after_time': after_time.isoformat(),
                    'time_between': str(time_between),
                    'samples_before': idx,
                    'samples_after': len(df_sorted) - idx
                }
                analysis['breakpoints'].append(break_info)

        # 统计断点分布
        if break_gaps.size > 0:
            analysis['min_break_gap'] = float(np.min(break_gaps))
            analysis['max_break_gap'] = float(np.max(break_gaps))
            analysis['avg_break_gap'] = float(np.mean(break_gaps))
            analysis['median_break_gap'] = float(np.median(break_gaps))

        return analysis

    def filter_segments_by_length(self, segments: List[pd.DataFrame],
                                 min_length: int = 10) -> List[pd.DataFrame]:
        """
        根据最小长度过滤连续段

        Args:
            segments: 连续段列表
            min_length: 最小长度阈值

        Returns:
            过滤后的连续段列表
        """
        filtered_segments = [seg for seg in segments if len(seg) >= min_length]

        removed_count = len(segments) - len(filtered_segments)
        removed_samples = sum(len(seg) for seg in segments) - sum(len(seg) for seg in filtered_segments)

        if removed_count > 0:
            logger.info(f"过滤掉 {removed_count} 个短段，共 {removed_samples} 个样本")

        return filtered_segments

if __name__ == "__main__":
    # 测试代码
    import yaml

    # 创建测试配置
    test_config = {
        'data_processing': {
            'max_time_gap': 500,
            'time_col': 'DATE'
        },
        'enhanced_processing': {
            'continuity_detection': True
        }
    }

    # 创建检测器
    detector = ContinuityDetector(test_config)

    print("连续段检测器测试")
    print("=" * 60)
    print(f"最大时间间隔阈值: {detector.max_time_gap}秒")
    print(f"时间列: {detector.time_col}")
    print(f"启用状态: {detector.enabled}")
    print("=" * 60)