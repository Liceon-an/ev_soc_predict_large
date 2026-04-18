#!/usr/bin/env python3
"""
特征工程模块
为电动汽车SOC预测任务计算衍生特征
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    特征工程类，负责计算各种衍生特征
    
    支持的特征：
    1. 平均加速度
    2. 速度分类（低/中/高，使用K-means聚类）
    3. 片段里程
    4. 速度占比（低/中/高速时间占比）
    5. 巡航时间比
    6. 瞬时功率
    7. 片段总能耗
    """
    
    def __init__(self, config: Dict):
        """
        初始化特征工程类
        
        Args:
            config: 配置字典，包含特征工程相关参数
        """
        self.config = config
        
        # 从配置中提取特征工程参数
        self.fe_config = config.get('feature_engineering', {})
        self.enabled = self.fe_config.get('enabled', False)
        
        # 速度聚类参数
        self.speed_cluster_config = self.fe_config.get('speed_clustering', {})
        self.use_clustering = self.speed_cluster_config.get('enabled', True)
        self.n_clusters = self.speed_cluster_config.get('n_clusters', 3)
        self.cluster_names = self.speed_cluster_config.get('cluster_names', ['low', 'medium', 'high'])
        
        # 加速度计算参数
        self.accel_config = self.fe_config.get('acceleration', {})
        self.min_time_diff = self.accel_config.get('min_time_diff', 0.1)  # 最小时间间隔
        self.max_acceleration = self.accel_config.get('max_acceleration', 10.0)  # 最大加速度限制
        
        # 功率计算参数
        self.power_config = self.fe_config.get('power', {})
        self.consider_charging = self.power_config.get('consider_charging', False)  # 是否考虑充电
        
        # 初始化聚类模型（延迟加载）
        self.kmeans_model = None
        self.speed_thresholds = None
        
        # 特征缓存
        self.cached_features = {}
        
        logger.info(f"特征工程初始化完成，启用状态: {self.enabled}")
        if self.enabled:
            logger.info(f"速度聚类: {'启用' if self.use_clustering else '禁用'}")
            logger.info(f"聚类数量: {self.n_clusters}")
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        在数据上拟合特征工程模型（如K-means聚类）
        
        Args:
            df: 训练数据
            
        Returns:
            self: 返回拟合后的特征工程对象
        """
        if not self.enabled:
            return self
        
        # 拟合速度聚类模型
        if self.use_clustering and 'speed' in df.columns:
            self._fit_speed_clustering(df['speed'].values)
        
        return self
    
    def _fit_speed_clustering(self, speeds: np.ndarray):
        """
        拟合速度聚类模型
        
        Args:
            speeds: 速度数组
        """
        logger.info("拟合速度聚类模型...")
        
        # 过滤异常值
        valid_speeds = speeds[~np.isnan(speeds)]
        valid_speeds = valid_speeds.reshape(-1, 1)
        
        if len(valid_speeds) < self.n_clusters:
            logger.warning(f"数据点数量({len(valid_speeds)})少于聚类数({self.n_clusters})，使用分位数阈值")
            self._set_quantile_thresholds(speeds)
            return
        
        try:
            # 使用K-means聚类
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            kmeans.fit(valid_speeds)
            
            # 获取聚类中心并排序
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            sorted_centers = centers[sorted_indices]
            
            # 计算阈值（聚类中心之间的中点）
            thresholds = []
            for i in range(len(sorted_centers) - 1):
                threshold = (sorted_centers[i] + sorted_centers[i + 1]) / 2
                thresholds.append(threshold)
            
            self.speed_thresholds = thresholds
            self.kmeans_model = kmeans
            
            logger.info(f"速度聚类完成，阈值: {thresholds}")
            logger.info(f"聚类中心: {sorted_centers}")
            
        except Exception as e:
            logger.warning(f"K-means聚类失败: {e}，使用分位数阈值")
            self._set_quantile_thresholds(speeds)
    
    def _set_quantile_thresholds(self, speeds: np.ndarray):
        """
        使用分位数设置速度阈值
        
        Args:
            speeds: 速度数组
        """
        valid_speeds = speeds[~np.isnan(speeds)]
        
        if len(valid_speeds) == 0:
            logger.warning("没有有效的速度数据，使用默认阈值")
            self.speed_thresholds = [20.0, 60.0]  # 默认阈值
            return
        
        # 根据聚类数量计算分位数
        if self.n_clusters == 3:
            # 三分位数
            thresholds = np.percentile(valid_speeds, [33.3, 66.7])
        elif self.n_clusters == 4:
            # 四分位数
            thresholds = np.percentile(valid_speeds, [25, 50, 75])
        else:
            # 默认三分位数
            thresholds = np.percentile(valid_speeds, [33.3, 66.7])
        
        self.speed_thresholds = thresholds.tolist()
        logger.info(f"使用分位数阈值: {self.speed_thresholds}")
    
    def compute_features(self, df: pd.DataFrame, is_sequence: bool = False) -> pd.DataFrame:
        """
        计算所有衍生特征
        
        Args:
            df: 输入数据框
            is_sequence: 是否为序列数据（时间窗口）
            
        Returns:
            添加了衍生特征的数据框
        """
        if not self.enabled:
            return df
        
        logger.debug(f"计算衍生特征，数据形状: {df.shape}, 序列模式: {is_sequence}")
        
        # 复制数据框以避免修改原始数据
        result_df = df.copy()
        
        # 计算基础衍生特征
        if 'speed' in df.columns and 'time_diff' in df.columns:
            result_df = self._add_acceleration_features(result_df)
        
        if 'speed' in df.columns:
            result_df = self._add_speed_category_features(result_df)
        
        if 'mileage' in df.columns:
            result_df = self._add_distance_features(result_df, is_sequence)
        
        if 'speed' in df.columns:
            result_df = self._add_speed_ratio_features(result_df)
        
        if 'speed' in df.columns:
            result_df = self._add_cruising_ratio_features(result_df)
        
        if 'total_volt' in df.columns and 'total_current' in df.columns:
            result_df = self._add_power_features(result_df)
        
        if 'total_volt' in df.columns and 'total_current' in df.columns and 'time_diff' in df.columns:
            result_df = self._add_energy_features(result_df, is_sequence)
        
        logger.debug(f"特征计算完成，新数据形状: {result_df.shape}")
        return result_df
    
    def _add_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加加速度相关特征"""
        result_df = df.copy()
        
        # 计算加速度 (km/h/s)
        speeds = df['speed'].values
        time_diffs = df['time_diff'].values
        
        # 处理边界条件
        accelerations = np.zeros_like(speeds, dtype=float)
        
        # 计算差分加速度
        for i in range(1, len(speeds)):
            if time_diffs[i] > self.min_time_diff:
                acceleration = (speeds[i] - speeds[i-1]) / time_diffs[i]
                # 过滤异常值
                if abs(acceleration) <= self.max_acceleration:
                    accelerations[i] = acceleration
        
        result_df['acceleration'] = accelerations
        
        # 计算平均加速度（绝对值）
        result_df['abs_acceleration'] = np.abs(accelerations)
        
        return result_df
    
    def _add_speed_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加速度分类特征"""
        result_df = df.copy()
        speeds = df['speed'].values
        
        if self.speed_thresholds is None:
            # 如果没有阈值，使用默认值
            thresholds = [20.0, 60.0]
        else:
            thresholds = self.speed_thresholds
        
        # 初始化分类数组
        speed_categories = np.zeros(len(speeds), dtype=int)
        
        if len(thresholds) == 2:
            # 3类分类：低、中、高
            for i, speed in enumerate(speeds):
                if np.isnan(speed):
                    speed_categories[i] = -1  # 无效值
                elif speed < thresholds[0]:
                    speed_categories[i] = 0  # 低速
                elif speed < thresholds[1]:
                    speed_categories[i] = 1  # 中速
                else:
                    speed_categories[i] = 2  # 高速
        elif len(thresholds) == 3:
            # 4类分类
            for i, speed in enumerate(speeds):
                if np.isnan(speed):
                    speed_categories[i] = -1
                elif speed < thresholds[0]:
                    speed_categories[i] = 0
                elif speed < thresholds[1]:
                    speed_categories[i] = 1
                elif speed < thresholds[2]:
                    speed_categories[i] = 2
                else:
                    speed_categories[i] = 3
        else:
            # 默认3类
            for i, speed in enumerate(speeds):
                if np.isnan(speed):
                    speed_categories[i] = -1
                elif speed < 20.0:
                    speed_categories[i] = 0
                elif speed < 60.0:
                    speed_categories[i] = 1
                else:
                    speed_categories[i] = 2
        
        result_df['speed_category'] = speed_categories
        
        return result_df
    
    def _add_distance_features(self, df: pd.DataFrame, is_sequence: bool) -> pd.DataFrame:
        """添加距离相关特征"""
        result_df = df.copy()
        
        if 'mileage' not in df.columns:
            return result_df
        
        mileages = df['mileage'].values
        
        if is_sequence and len(mileages) > 0:
            # 对于序列数据，计算总行驶距离
            total_distance = mileages[-1] - mileages[0]
            result_df['segment_distance'] = total_distance
        else:
            # 对于单点数据，计算与前一时刻的距离差
            distances = np.zeros_like(mileages, dtype=float)
            for i in range(1, len(mileages)):
                distances[i] = mileages[i] - mileages[i-1]
            result_df['instant_distance'] = distances
        
        return result_df
    
    def _add_speed_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加速度占比特征"""
        result_df = df.copy()
        
        if 'speed_category' not in df.columns:
            # 如果没有速度分类，先计算
            result_df = self._add_speed_category_features(result_df)
        
        speed_categories = result_df['speed_category'].values
        
        # 计算各类别的占比
        total_points = len(speed_categories)
        valid_mask = speed_categories >= 0  # 排除无效值
        
        if total_points == 0:
            return result_df
        
        # 计算各类别数量
        if self.n_clusters == 3:
            low_count = np.sum((speed_categories == 0) & valid_mask)
            medium_count = np.sum((speed_categories == 1) & valid_mask)
            high_count = np.sum((speed_categories == 2) & valid_mask)
            
            result_df['low_speed_ratio'] = low_count / total_points
            result_df['medium_speed_ratio'] = medium_count / total_points
            result_df['high_speed_ratio'] = high_count / total_points
            
        elif self.n_clusters == 4:
            low_count = np.sum((speed_categories == 0) & valid_mask)
            medium_low_count = np.sum((speed_categories == 1) & valid_mask)
            medium_high_count = np.sum((speed_categories == 2) & valid_mask)
            high_count = np.sum((speed_categories == 3) & valid_mask)
            
            result_df['low_speed_ratio'] = low_count / total_points
            result_df['medium_low_speed_ratio'] = medium_low_count / total_points
            result_df['medium_high_speed_ratio'] = medium_high_count / total_points
            result_df['high_speed_ratio'] = high_count / total_points
        
        return result_df
    
    def _add_cruising_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加巡航时间比特征"""
        result_df = df.copy()
        
        if 'speed' not in df.columns:
            return result_df
        
        speeds = df['speed'].values
        
        # 判断是否在行驶（速度 > 0.1 km/h 视为行驶）
        is_moving = speeds > 0.1
        cruising_ratio = np.mean(is_moving) if len(speeds) > 0 else 0.0
        
        result_df['cruising_ratio'] = cruising_ratio
        
        # 计算连续行驶时间
        if len(speeds) > 1:
            # 找到行驶段的开始和结束
            moving_segments = []
            in_segment = False
            segment_start = 0
            
            for i in range(len(speeds)):
                if is_moving[i] and not in_segment:
                    in_segment = True
                    segment_start = i
                elif not is_moving[i] and in_segment:
                    in_segment = False
                    moving_segments.append((segment_start, i-1))
            
            # 处理最后一个段
            if in_segment:
                moving_segments.append((segment_start, len(speeds)-1))
            
            # 计算平均行驶段长度
            if moving_segments:
                segment_lengths = [end - start + 1 for start, end in moving_segments]
                result_df['avg_moving_segment_length'] = np.mean(segment_lengths)
            else:
                result_df['avg_moving_segment_length'] = 0.0
        
        return result_df
    
    def _add_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加功率相关特征"""
        result_df = df.copy()
        
        voltages = df['total_volt'].values
        currents = df['total_current'].values
        
        # 计算瞬时功率 (W)
        instant_power = voltages * currents
        
        # 转换为 kW
        instant_power_kw = instant_power / 1000.0
        
        result_df['instant_power_w'] = instant_power
        result_df['instant_power_kw'] = instant_power_kw
        
        # 判断功率方向（正为放电，负为充电）
        power_direction = np.sign(instant_power)
        result_df['power_direction'] = power_direction
        
        # 计算有效功率（只考虑放电）
        if not self.consider_charging:
            effective_power = np.where(instant_power > 0, instant_power, 0)
            result_df['effective_power_w'] = effective_power
            result_df['effective_power_kw'] = effective_power / 1000.0
        
        return result_df
    
    def _add_energy_features(self, df: pd.DataFrame, is_sequence: bool) -> pd.DataFrame:
        """添加能量相关特征"""
        result_df = df.copy()
        
        if 'instant_power_w' not in df.columns:
            result_df = self._add_power_features(result_df)
        
        powers = result_df['instant_power_w'].values
        time_diffs = df['time_diff'].values
        
        # 计算能量 (J = W * s)
        energy_joules = powers * time_diffs
        
        # 转换为 kWh
        energy_kwh = energy_joules / (1000.0 * 3600.0)
        
        result_df['instant_energy_j'] = energy_joules
        result_df['instant_energy_kwh'] = energy_kwh
        
        if is_sequence and len(powers) > 0:
            # 对于序列数据，计算总能量
            total_energy_j = np.sum(energy_joules)
            total_energy_kwh = np.sum(energy_kwh)
            
            result_df['total_energy_j'] = total_energy_j
            result_df['total_energy_kwh'] = total_energy_kwh
            
            # 计算有效能量（只考虑放电）
            if not self.consider_charging and 'effective_power_w' in result_df.columns:
                effective_powers = result_df['effective_power_w'].values
                effective_energy_j = effective_powers * time_diffs
                effective_energy_kwh = effective_energy_j / (1000.0 * 3600.0)
                
                result_df['effective_energy_j'] = np.sum(effective_energy_j)
                result_df['effective_energy_kwh'] = np.sum(effective_energy_kwh)
        
        return result_df
    
    def get_feature_names(self, original_features: List[str]) -> List[str]:
        """
        获取所有特征名称（包括原始特征和衍生特征）
        
        Args:
            original_features: 原始特征名称列表
            
        Returns:
            所有特征名称列表
        """
        if not self.enabled:
            return original_features
        
        # 衍生特征名称
        derived_features = [
            'acceleration',
            'abs_acceleration',
            'speed_category',
            'segment_distance',
            'low_speed_ratio',
            'medium_speed_ratio',
            'high_speed_ratio',
            'cruising_ratio',
            'avg_moving_segment_length',
            'instant_power_w',
            'instant_power_kw',
            'power_direction',
            'instant_energy_j',
            'instant_energy_kwh',
            'total_energy_j',
            'total_energy_kwh'
        ]
        
        # 根据配置添加可选特征
        if not self.consider_charging:
            derived_features.extend(['effective_power_w', 'effective_power_kw'])
            derived_features.extend(['effective_energy_j', 'effective_energy_kwh'])
        
        # 合并特征列表
        all_features = original_features + derived_features
        
        # 去除可能重复的特征
        unique_features = []
        for feature in all_features:
            if feature not in unique_features:
                unique_features.append(feature)
        
        return unique_features
    
    def get_config(self) -> Dict:
        """
        获取特征工程配置
        
        Returns:
            特征工程配置字典
        """
        config = {
            'enabled': self.enabled,
            'speed_clustering': {
                'enabled': self.use_clustering,
                'n_clusters': self.n_clusters,
                'cluster_names': self.cluster_names,
                'thresholds': self.speed_thresholds
            },
            'acceleration': {
                'min_time_diff': self.min_time_diff,
                'max_acceleration': self.max_acceleration
            },
            'power': {
                'consider_charging': self.consider_charging
            }
        }
        
        return config

# 工具函数
def create_feature_engineering_config(
    enabled: bool = True,
    use_speed_clustering: bool = True,
    n_clusters: int = 3,
    consider_charging: bool = False
) -> Dict:
    """
    创建特征工程配置
    
    Args:
        enabled: 是否启用特征工程
        use_speed_clustering: 是否使用速度聚类
        n_clusters: 聚类数量
        consider_charging: 是否考虑充电功率
        
    Returns:
        特征工程配置字典
    """
    config = {
        'feature_engineering': {
            'enabled': enabled,
            'speed_clustering': {
                'enabled': use_speed_clustering,
                'n_clusters': n_clusters,
                'cluster_names': ['low', 'medium', 'high'][:n_clusters]
            },
            'acceleration': {
                'min_time_diff': 0.1,
                'max_acceleration': 10.0
            },
            'power': {
                'consider_charging': consider_charging
            }
        }
    }
    
    return config

if __name__ == "__main__":
    # 测试代码
    import yaml
    
    # 创建测试配置
    test_config = create_feature_engineering_config()
    
    # 创建特征工程对象
    engineer = FeatureEngineer(test_config)
    
    print("特征工程模块测试")
    print("=" * 60)
    print(f"启用状态: {engineer.enabled}")
    print(f"速度聚类: {engineer.use_clustering}")
    print(f"聚类数量: {engineer.n_clusters}")
    print("=" * 60)
