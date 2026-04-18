#!/usr/bin/env python3
"""
增强特征工程模块
处理独立的时间片段，支持序列级特征计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.cluster import KMeans
import warnings

logger = logging.getLogger(__name__)

def process_time_segment(segment_df: pd.DataFrame, feature_config: Dict,
                        global_config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    处理单个时间片段，计算所有特征

    Args:
        segment_df: 时间片段数据框（包含上下文）
        feature_config: 特征工程配置
        global_config: 全局配置

    Returns:
        (特征数组, 标签数组, 特征名称列表)
    """
    if not feature_config.get('enabled', False):
        logger.debug("特征工程已禁用，返回原始特征")
        return _extract_original_features(segment_df, global_config)

    # 提取核心序列（去掉上下文）
    core_segment = segment_df[segment_df['is_core']].copy()
    if len(core_segment) == 0:
        raise ValueError("片段中没有核心序列")

    # 获取特征类型配置
    feature_types = feature_config.get('feature_types', {})
    use_point_features = feature_types.get('point_features', True)
    use_sequence_features = feature_types.get('sequence_features', True)
    use_boundary_features = feature_types.get('boundary_features', True)

    # 计算特征
    features_dict = {}

    # 1. 单点特征（每个时间点的特征）
    if use_point_features:
        point_features = _compute_point_features(core_segment, feature_config)
        features_dict.update(point_features)

    # 2. 序列级特征（整个序列的统计特征）
    if use_sequence_features:
        sequence_features = _compute_sequence_features(core_segment, feature_config)
        features_dict.update(sequence_features)

    # 3. 边界特征（利用上下文计算的边界特征）
    if use_boundary_features and 'is_context' in segment_df.columns:
        boundary_features = _compute_boundary_features(segment_df, feature_config)
        features_dict.update(boundary_features)

    # 4. 原始特征
    original_features = _extract_original_features_dict(core_segment, global_config)
    features_dict.update(original_features)

    # 转换为数组
    feature_names = list(features_dict.keys())
    feature_values = list(features_dict.values())

    # 确保所有特征值都是数值类型
    feature_array = np.array(feature_values, dtype=np.float32).reshape(1, -1)

    # 提取标签
    label_col = global_config.get('data_processing', {}).get('label_col', 'refined_soc')
    if label_col in core_segment.columns:
        labels = core_segment[label_col].values
        label_value = np.mean(labels)  # 使用序列平均值作为标签
    else:
        logger.warning(f"标签列 '{label_col}' 不存在，使用0作为默认值")
        label_value = 0.0

    label_array = np.array([label_value], dtype=np.float32)

    logger.debug(f"为片段生成 {len(feature_names)} 个特征")
    return feature_array, label_array, feature_names

def _extract_original_features(segment_df: pd.DataFrame, global_config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    提取原始特征（特征工程禁用时使用）

    Args:
        segment_df: 时间片段数据框
        global_config: 全局配置

    Returns:
        (特征数组, 标签数组, 特征名称列表)
    """
    core_segment = segment_df[segment_df['is_core']].copy()

    # 获取特征列
    feature_cols = global_config.get('data_processing', {}).get('feature_cols', [])
    available_cols = [col for col in feature_cols if col in core_segment.columns]

    if not available_cols:
        raise ValueError("没有可用的特征列")

    # 提取特征
    features = core_segment[available_cols].values.T  # 转置为 (特征数, 序列长度)

    # 提取标签
    label_col = global_config.get('data_processing', {}).get('label_col', 'refined_soc')
    if label_col in core_segment.columns:
        labels = core_segment[label_col].values
        label_value = np.mean(labels)
    else:
        label_value = 0.0

    label_array = np.array([label_value], dtype=np.float32)

    return features, label_array, available_cols

def _extract_original_features_dict(core_segment: pd.DataFrame, global_config: Dict) -> Dict[str, float]:
    """
    提取原始特征并返回字典格式

    Args:
        core_segment: 核心序列数据框
        global_config: 全局配置

    Returns:
        特征字典
    """
    features_dict = {}

    # 获取特征列
    feature_cols = global_config.get('data_processing', {}).get('feature_cols', [])
    base_features = global_config.get('feature_engineering', {}).get('base_features', [])

    # 优先使用base_features，如果不存在则使用feature_cols
    use_cols = base_features if base_features else feature_cols
    available_cols = [col for col in use_cols if col in core_segment.columns]

    # 计算每个特征的统计量（平均值）
    for col in available_cols:
        if col in core_segment.columns:
            features_dict[f"{col}_mean"] = float(core_segment[col].mean())
            features_dict[f"{col}_std"] = float(core_segment[col].std())
            features_dict[f"{col}_min"] = float(core_segment[col].min())
            features_dict[f"{col}_max"] = float(core_segment[col].max())

    return features_dict

def _compute_point_features(core_segment: pd.DataFrame, feature_config: Dict) -> Dict[str, float]:
    """
    计算单点特征

    Args:
        core_segment: 核心序列数据框
        feature_config: 特征工程配置

    Returns:
        单点特征字典
    """
    features_dict = {}

    # 1. 速度相关特征
    if 'speed' in core_segment.columns:
        speeds = core_segment['speed'].values

        # 速度统计
        features_dict['speed_mean'] = float(np.mean(speeds))
        features_dict['speed_std'] = float(np.std(speeds))
        features_dict['speed_min'] = float(np.min(speeds))
        features_dict['speed_max'] = float(np.max(speeds))

        # 速度分类（使用K-means或分位数）
        speed_thresholds = _get_speed_thresholds(speeds, feature_config)
        if speed_thresholds:
            if len(speed_thresholds) == 1:
                low_mask = speeds <= speed_thresholds[0]
                high_mask = speeds > speed_thresholds[0]
                features_dict['low_speed_ratio'] = float(np.mean(low_mask))
                features_dict['high_speed_ratio'] = float(np.mean(high_mask))
            elif len(speed_thresholds) >= 2:
                low_mask = speeds <= speed_thresholds[0]
                medium_mask = (speeds > speed_thresholds[0]) & (speeds <= speed_thresholds[1])
                high_mask = speeds > speed_thresholds[1]
                features_dict['low_speed_ratio'] = float(np.mean(low_mask))
                features_dict['medium_speed_ratio'] = float(np.mean(medium_mask))
                features_dict['high_speed_ratio'] = float(np.mean(high_mask))

    # 2. 加速度特征
    if 'speed' in core_segment.columns and 'time_diff' in core_segment.columns:
        speeds = core_segment['speed'].values
        time_diffs = core_segment['time_diff'].values

        # 计算加速度
        valid_indices = time_diffs > 0.1  # 最小时间间隔
        if np.any(valid_indices):
            accels = np.zeros_like(speeds)
            accels[1:] = (speeds[1:] - speeds[:-1]) / time_diffs[1:]

            # 限制加速度范围
            max_accel = feature_config.get('acceleration', {}).get('max_acceleration', 10.0)
            accels = np.clip(accels, -max_accel, max_accel)

            features_dict['acceleration_mean'] = float(np.mean(accels[valid_indices]))
            features_dict['acceleration_std'] = float(np.std(accels[valid_indices]))
            features_dict['acceleration_abs_mean'] = float(np.mean(np.abs(accels[valid_indices])))

    # 3. 功率特征
    if 'total_volt' in core_segment.columns and 'total_current' in core_segment.columns:
        voltages = core_segment['total_volt'].values
        currents = core_segment['total_current'].values

        # 瞬时功率
        instant_power = voltages * currents  # W

        features_dict['instant_power_mean'] = float(np.mean(instant_power))
        features_dict['instant_power_std'] = float(np.std(instant_power))
        features_dict['instant_power_min'] = float(np.min(instant_power))
        features_dict['instant_power_max'] = float(np.max(instant_power))

        # 功率方向
        power_direction = np.sign(instant_power)
        features_dict['discharge_ratio'] = float(np.mean(power_direction > 0))
        features_dict['charge_ratio'] = float(np.mean(power_direction < 0))

    return features_dict

def _compute_sequence_features(core_segment: pd.DataFrame, feature_config: Dict) -> Dict[str, float]:
    """
    计算序列级特征

    Args:
        core_segment: 核心序列数据框
        feature_config: 特征工程配置

    Returns:
        序列特征字典
    """
    features_dict = {}

    # 1. 里程特征
    if 'mileage' in core_segment.columns:
        mileage = core_segment['mileage'].values
        if len(mileage) > 1:
            segment_distance = mileage[-1] - mileage[0]
            features_dict['segment_distance'] = float(segment_distance)

            # 平均速度（如果有时差）
            if 'time_diff' in core_segment.columns:
                total_time = core_segment['time_diff'].sum()
                if total_time > 0:
                    avg_speed = segment_distance / total_time
                    features_dict['segment_avg_speed'] = float(avg_speed)

    # 2. 能量特征
    if 'total_volt' in core_segment.columns and 'total_current' in core_segment.columns:
        voltages = core_segment['total_volt'].values
        currents = core_segment['total_current'].values
        time_diffs = core_segment['time_diff'].values if 'time_diff' in core_segment.columns else np.ones(len(voltages))

        # 计算能量
        instant_power = voltages * currents  # W
        instant_energy = instant_power * time_diffs  # J

        total_energy_j = np.sum(instant_energy)
        total_energy_kwh = total_energy_j / (1000.0 * 3600.0)

        features_dict['total_energy_j'] = float(total_energy_j)
        features_dict['total_energy_kwh'] = float(total_energy_kwh)

        # 有效能量（只考虑放电）
        consider_charging = feature_config.get('power', {}).get('consider_charging', False)
        if not consider_charging:
            discharge_mask = instant_power > 0
            if np.any(discharge_mask):
                discharge_energy = np.sum(instant_energy[discharge_mask])
                features_dict['discharge_energy_j'] = float(discharge_energy)
                features_dict['discharge_energy_kwh'] = float(discharge_energy / (1000.0 * 3600.0))

    # 3. 巡航特征
    if 'speed' in core_segment.columns:
        speeds = core_segment['speed'].values

        # 巡航状态（速度变化小）
        if len(speeds) > 1:
            speed_diffs = np.abs(np.diff(speeds))
            cruising_mask = speed_diffs < 1.0  # 速度变化小于1 km/h
            cruising_ratio = np.mean(cruising_mask) if len(cruising_mask) > 0 else 0.0
            features_dict['cruising_ratio'] = float(cruising_ratio)

            # 平均巡航段长度
            if np.any(cruising_mask):
                # 计算连续巡航段长度
                cruising_segments = []
                current_length = 0
                for is_cruising in cruising_mask:
                    if is_cruising:
                        current_length += 1
                    else:
                        if current_length > 0:
                            cruising_segments.append(current_length)
                            current_length = 0
                if current_length > 0:
                    cruising_segments.append(current_length)

                if cruising_segments:
                    features_dict['avg_cruising_segment_length'] = float(np.mean(cruising_segments))

    return features_dict

def _compute_boundary_features(full_segment: pd.DataFrame, feature_config: Dict) -> Dict[str, float]:
    """
    计算边界特征（利用上下文）

    Args:
        full_segment: 完整片段数据框（包含上下文）
        feature_config: 特征工程配置

    Returns:
        边界特征字典
    """
    features_dict = {}

    if 'is_context' not in full_segment.columns or 'is_core' not in full_segment.columns:
        return features_dict

    # 提取上下文和核心区域
    context_segment = full_segment[full_segment['is_context']].copy()
    core_segment = full_segment[full_segment['is_core']].copy()

    if len(context_segment) == 0 or len(core_segment) == 0:
        return features_dict

    # 1. 边界速度特征
    if 'speed' in full_segment.columns:
        # 前后上下文的速度
        before_context = context_segment[context_segment.index < core_segment.index[0]]
        after_context = context_segment[context_segment.index > core_segment.index[-1]]

        if len(before_context) > 0:
            features_dict['speed_before_mean'] = float(before_context['speed'].mean())
        if len(after_context) > 0:
            features_dict['speed_after_mean'] = float(after_context['speed'].mean())

        # 边界速度变化
        if len(before_context) > 0 and len(core_segment) > 0:
            speed_before = before_context['speed'].iloc[-1] if len(before_context) > 0 else core_segment['speed'].iloc[0]
            speed_start = core_segment['speed'].iloc[0]
            features_dict['speed_boundary_change_start'] = float(speed_start - speed_before)

        if len(after_context) > 0 and len(core_segment) > 0:
            speed_end = core_segment['speed'].iloc[-1]
            speed_after = after_context['speed'].iloc[0] if len(after_context) > 0 else speed_end
            features_dict['speed_boundary_change_end'] = float(speed_after - speed_end)

    # 2. 边界加速度特征
    if 'speed' in full_segment.columns and 'time_diff' in full_segment.columns:
        # 找到核心序列的边界点
        core_start_idx = core_segment.index[0]
        core_end_idx = core_segment.index[-1]

        # 获取边界点附近的数据
        boundary_points = []
        if core_start_idx > 0:
            boundary_points.append(core_start_idx - 1)  # 核心序列前一点
        boundary_points.append(core_start_idx)  # 核心序列第一点
        boundary_points.append(core_end_idx)  # 核心序列最后一点
        if core_end_idx < len(full_segment) - 1:
            boundary_points.append(core_end_idx + 1)  # 核心序列后一点

        if len(boundary_points) >= 2:
            boundary_df = full_segment.loc[boundary_points]
            speeds = boundary_df['speed'].values
            time_diffs = boundary_df['time_diff'].values

            # 计算边界加速度
            if len(speeds) > 1 and np.all(time_diffs[1:] > 0.1):
                boundary_accels = (speeds[1:] - speeds[:-1]) / time_diffs[1:]
                if len(boundary_accels) > 0:
                    features_dict['boundary_acceleration_mean'] = float(np.mean(boundary_accels))
                    features_dict['boundary_acceleration_max'] = float(np.max(np.abs(boundary_accels)))

    return features_dict

def _get_speed_thresholds(speeds: np.ndarray, feature_config: Dict) -> List[float]:
    """
    获取速度分类阈值

    Args:
        speeds: 速度数组
        feature_config: 特征工程配置

    Returns:
        速度阈值列表
    """
    speed_config = feature_config.get('speed_clustering', {})
    use_clustering = speed_config.get('enabled', True)
    n_clusters = speed_config.get('n_clusters', 3)

    valid_speeds = speeds[~np.isnan(speeds)]

    if len(valid_speeds) < n_clusters:
        # 数据不足，使用分位数
        return _get_quantile_thresholds(valid_speeds, n_clusters)

    if use_clustering:
        try:
            # 使用K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(valid_speeds.reshape(-1, 1))

            # 获取聚类中心并排序
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            sorted_centers = centers[sorted_indices]

            # 计算阈值（聚类中心之间的中点）
            thresholds = []
            for i in range(len(sorted_centers) - 1):
                threshold = (sorted_centers[i] + sorted_centers[i + 1]) / 2
                thresholds.append(threshold)

            return thresholds

        except Exception as e:
            logger.warning(f"K-means聚类失败: {e}，使用分位数阈值")
            return _get_quantile_thresholds(valid_speeds, n_clusters)
    else:
        # 使用配置中的固定阈值
        fixed_thresholds = feature_config.get('speed_thresholds', {})
        if fixed_thresholds:
            thresholds = []
            if 'low' in fixed_thresholds:
                thresholds.append(fixed_thresholds['low'])
            if 'medium' in fixed_thresholds:
                thresholds.append(fixed_thresholds['medium'])
            return thresholds
        else:
            # 使用分位数
            return _get_quantile_thresholds(valid_speeds, n_clusters)

def _get_quantile_thresholds(speeds: np.ndarray, n_clusters: int) -> List[float]:
    """
    使用分位数获取速度阈值

    Args:
        speeds: 速度数组
        n_clusters: 聚类数量

    Returns:
        速度阈值列表
    """
    if len(speeds) == 0:
        return []

    if n_clusters == 2:
        # 二分位数（中位数）
        threshold = np.percentile(speeds, 50)
        return [threshold]
    elif n_clusters == 3:
        # 三分位数
        thresholds = np.percentile(speeds, [33.3, 66.7])
        return thresholds.tolist()
    elif n_clusters == 4:
        # 四分位数
        thresholds = np.percentile(speeds, [25, 50, 75])
        return thresholds.tolist()
    else:
        # 默认三分位数
        thresholds = np.percentile(speeds, [33.3, 66.7])
        return thresholds.tolist()

# 测试函数
def test_process_time_segment():
    """测试process_time_segment函数"""
    # 创建测试数据
    n_samples = 44  # 40核心 + 2×2上下文
    test_data = {
        'speed': np.random.uniform(0, 100, n_samples),
        'mileage': np.cumsum(np.random.uniform(0, 0.1, n_samples)),
        'total_volt': np.random.uniform(300, 400, n_samples),
        'total_current': np.random.uniform(-100, 100, n_samples),
        'refined_soc': np.random.uniform(20, 80, n_samples),
        'time_diff': np.full(n_samples, 10.0),  # 10秒间隔
        'is_context': np.concatenate([np.ones(2), np.zeros(40), np.ones(2)]).astype(bool),
        'is_core': np.concatenate([np.zeros(2), np.ones(40), np.zeros(2)]).astype(bool)
    }

    test_df = pd.DataFrame(test_data)

    # 创建测试配置
    test_config = {
        'feature_engineering': {
            'enabled': True,
            'feature_types': {
                'point_features': True,
                'sequence_features': True,
                'boundary_features': True
            },
            'speed_clustering': {
                'enabled': True,
                'n_clusters': 3
            },
            'acceleration': {
                'max_acceleration': 10.0
            },
            'power': {
                'consider_charging': False
            }
        },
        'data_processing': {
            'feature_cols': ['speed', 'mileage', 'total_volt', 'total_current'],
            'label_col': 'refined_soc'
        }
    }

    # 处理片段
    features, labels, feature_names = process_time_segment(test_df, test_config['feature_engineering'], test_config)

    print("测试结果:")
    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"前10个特征名: {feature_names[:10]}")

    return features, labels, feature_names

if __name__ == "__main__":
    # 运行测试
    print("增强特征工程模块测试")
    print("=" * 60)
    features, labels, feature_names = test_process_time_segment()
    print("=" * 60)
    print("测试完成！")