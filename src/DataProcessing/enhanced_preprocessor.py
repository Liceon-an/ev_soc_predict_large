#!/usr/bin/env python3
"""
增强数据预处理器
新架构的主处理流程，协调连续段检测、时间片段划分和特征工程
"""

import numpy as np
import pandas as pd
import yaml
import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.DataProcessing.continuity_detector import ContinuityDetector
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
from src.DataProcessing.time_segment_splitter import TimeSegmentSplitter
from src.FeatureEngineering.enhanced_feature_engineer import process_time_segment


logger = logging.getLogger(__name__)

class EnhancedDataPreparator:
    """
    增强数据预处理器类
    协调完整的新架构处理流程
    """

    def __init__(self, config: Dict):
        """
        初始化增强数据预处理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.experiment_name = config.get('experiment_name', 'enhanced_experiment')

        # 初始化组件
        self.continuity_detector = ContinuityDetector(config)
        self.time_segment_splitter = TimeSegmentSplitter(config)

        # 处理模式
        self.processing_mode = config.get('processing_mode', 'enhanced')
        self.feature_engineering_enabled = config.get('feature_engineering', {}).get('enabled', False)

        logger.info(f"增强数据预处理器初始化完成")
        logger.info(f"实验名称: {self.experiment_name}")
        logger.info(f"处理模式: {self.processing_mode}")
        logger.info(f"特征工程: {'启用' if self.feature_engineering_enabled else '禁用'}")

    def load_data(self, input_file: str) -> pd.DataFrame:
        """
        加载原始数据

        Args:
            input_file: 输入文件路径

        Returns:
            加载的数据框
        """
        logger.info(f"加载数据: {input_file}")

        try:
            # 根据文件扩展名选择加载方式
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.parquet'):
                df = pd.read_parquet(input_file)
            else:
                raise ValueError(f"不支持的文件格式: {input_file}")

            logger.info(f"数据加载成功，形状: {df.shape}")
            logger.info(f"数据列: {list(df.columns)}")

            # 检查必要的列
            required_cols = self.config.get('data_processing', {}).get('feature_cols', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")

            return df

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据（排序、计算时间间隔等）

        Args:
            df: 原始数据框

        Returns:
            准备好的数据框
        """
        logger.info("准备数据...")

        # 复制数据以避免修改原始数据
        prepared_df = df.copy()

        # 确保按时间排序
        time_col = self.config.get('data_processing', {}).get('time_col', 'DATE')
        if time_col in prepared_df.columns:
            prepared_df = prepared_df.sort_values(time_col).reset_index(drop=True)

            # 计算时间间隔
            times = pd.to_datetime(prepared_df[time_col])
            time_diffs = times.diff().dt.total_seconds()
            time_diffs.iloc[0] = 0  # 第一行设为0
            prepared_df['time_diff'] = time_diffs

            logger.info(f"时间间隔统计 - 最小值: {time_diffs.min():.2f}s, "
                       f"最大值: {time_diffs.max():.2f}s, "
                       f"平均值: {time_diffs.mean():.2f}s")

        return prepared_df

    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        执行完整的数据处理流程

        Args:
            df: 输入数据框

        Returns:
            (特征数组, 标签数组, 元数据字典)
        """
        logger.info("开始增强数据处理流程...")

        # 1. 准备数据
        prepared_df = self.prepare_data(df)

        # 2. 检测连续段
        logger.info("检测连续段...")
        continuous_segments = self.continuity_detector.detect_continuous_segments(prepared_df)

        # 获取连续段统计
        segment_stats = self.continuity_detector.get_segment_statistics(continuous_segments)
        logger.info(f"连续段统计: {segment_stats}")

        # 3. 划分时间片段
        logger.info("划分时间片段...")
        time_segments = self.time_segment_splitter.split_multiple_segments(continuous_segments)

        # 获取片段统计
        time_segment_stats = self.time_segment_splitter.get_segment_statistics(time_segments)
        logger.info(f"时间片段统计: {time_segment_stats}")

        # 4. 特征工程
        features_list = []
        labels_list = []
        feature_names = None

        if self.feature_engineering_enabled:
            logger.info("执行特征工程...")
            feature_config = self.config.get('feature_engineering', {})

            for i, segment in enumerate(time_segments):
                if i % 100 == 0:
                    logger.debug(f"处理第 {i+1}/{len(time_segments)} 个片段")

                try:
                    # 处理单个时间片段
                    segment_features, segment_labels, segment_feature_names = process_time_segment(
                        segment, feature_config, self.config
                    )

                    features_list.append(segment_features)
                    labels_list.append(segment_labels)

                    if feature_names is None:
                        feature_names = segment_feature_names

                except Exception as e:
                    logger.warning(f"处理片段 {i} 失败: {e}")
                    continue

        else:
            logger.info("特征工程已禁用，使用原始特征")
            # 提取原始特征和标签
            feature_cols = self.config.get('data_processing', {}).get('feature_cols', [])
            label_col = self.config.get('data_processing', {}).get('label_col', 'refined_soc')

            for segment in time_segments:
                # 只使用核心序列部分
                core_segment = segment[segment['is_core']]

                if len(core_segment) == self.time_segment_splitter.sequence_length:
                    segment_features = core_segment[feature_cols].values
                    segment_labels = core_segment[label_col].values

                    features_list.append(segment_features)
                    labels_list.append(segment_labels)

            feature_names = feature_cols

        # 转换为数组
        if not features_list:
            raise ValueError("没有生成任何有效特征")

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"特征数组形状: {X.shape}")
        logger.info(f"标签数组形状: {y.shape}")

        # 5. 收集元数据
        metadata = {
            'experiment_name': self.experiment_name,
            'processing_mode': self.processing_mode,
            'feature_engineering_enabled': self.feature_engineering_enabled,
            'total_samples': len(X),
            'feature_dim': X.shape[1],
            'sequence_length': X.shape[2],
            'feature_names': feature_names,
            'segment_statistics': segment_stats,
            'time_segment_statistics': time_segment_stats,
            'config': self.config,
            'processing_timestamp': datetime.now().isoformat()
        }

        return X, y, metadata

    def save_processed_data(self, X: np.ndarray, y: np.ndarray, metadata: Dict,
                           output_dir: str = 'data/processed'):
        """
        保存处理后的数据

        Args:
            X: 特征数组
            y: 标签数组
            metadata: 元数据字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.experiment_name}_{timestamp}"

        # 保存数据
        data_file = os.path.join(output_dir, f"{base_name}_data.npz")
        np.savez_compressed(data_file, X=X, y=y)

        # 保存元数据
        metadata_file = os.path.join(output_dir, f"{base_name}_metadata.yaml")
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # 保存特征名称
        if 'feature_names' in metadata:
            feature_file = os.path.join(output_dir, f"{base_name}_features.txt")
            with open(feature_file, 'w') as f:
                for feature in metadata['feature_names']:
                    f.write(f"{feature}\n")

        logger.info(f"数据保存完成:")
        logger.info(f"  数据文件: {data_file}")
        logger.info(f"  元数据: {metadata_file}")
        if 'feature_names' in metadata:
            logger.info(f"  特征文件: {feature_file}")

        return {
            'data_file': data_file,
            'metadata_file': metadata_file,
            'feature_file': feature_file if 'feature_names' in metadata else None
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强数据预处理器')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--input', type=str,
                       help='输入文件路径（覆盖配置文件中的设置）')
    parser.add_argument('--output', type=str, default='data/processed/enhanced',
                       help='输出目录')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level)

    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)

        # 创建预处理器
        preparator = EnhancedDataPreparator(config)

        # 确定输入文件
        input_file = args.input
        if not input_file:
            input_file = config.get('data_processing', {}).get('input_file')
            if not input_file:
                raise ValueError("未指定输入文件")

        # 添加数据目录前缀
        if not os.path.isabs(input_file):
            input_file = os.path.join('data', input_file)

        # 加载数据
        df = preparator.load_data(input_file)

        # 处理数据
        X, y, metadata = preparator.process(df)

        # 保存数据
        saved_files = preparator.save_processed_data(X, y, metadata, args.output)

        logger.info("增强数据处理完成！")
        logger.info(f"生成 {len(X)} 个样本")
        logger.info(f"特征维度: {X.shape[1]} × {X.shape[2]}")
        logger.info(f"输出文件保存在: {args.output}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()