#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_feature.py - 原始数据统计特征计算模块

功能: 对原始数据集计算统计信息，生成特征数据文件
输入: aligned_data_refined_soc.csv (原始数据)
输出: feature_data.csv (特征数据)

模块结构:
1. 数据加载与选择模块
2. 数据预处理模块
3. 数据判断模块
4. 差值计算模块
5. 特征乘积模块
6. 滑动窗口统计模块
7. 滑动窗口占比模块
8. 特征整合与列删除模块
9. 输出模块
10. 主函数
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
import os
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 数据加载与选择模块
# ============================================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    加载原始CSV数据

    参数:
        file_path: CSV文件路径

    返回:
        pandas DataFrame

    注意: 根据实际数据格式调整读取参数
    """
    try:
        logger.info(f"开始加载数据: {file_path}")
        # TODO: 根据实际数据格式调整读取参数
        # 示例: df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = pd.read_csv(file_path)
        logger.info(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


def select_data_rows(df: pd.DataFrame, n_rows: Optional[int] = None) -> pd.DataFrame:
    """
    选择处理的行数（前n行或全部）

    参数:
        df: 原始DataFrame
        n_rows: 可选，处理的行数限制，None表示全部

    返回:
        选择后的DataFrame
    """
    if n_rows is not None and n_rows > 0:
        logger.info(f"选择前 {n_rows} 行数据进行处理")
        return df.head(n_rows).copy()
    else:
        logger.info("处理全部数据")
        return df.copy()


# ============================================================================
# 2. 数据预处理模块
# ============================================================================

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    验证数据质量，返回质量报告

    参数:
        df: 要验证的DataFrame

    返回:
        质量报告字典，包含:
        - missing_values: 各列缺失值数量
        - data_types: 各列数据类型
        - basic_stats: 基础统计信息
        - warnings: 质量警告列表
    """
    logger.info("开始数据质量验证")

    report = {
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'basic_stats': {},
        'warnings': []
    }

    # 检查缺失值
    missing_cols = [col for col, count in report['missing_values'].items() if count > 0]
    if missing_cols:
        warning = f"发现缺失值的列: {missing_cols}"
        report['warnings'].append(warning)
        logger.warning(warning)

    # 基础统计信息（数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['basic_stats'] = df[numeric_cols].describe().to_dict()

    logger.info("数据质量验证完成")
    return report


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    数据预处理

    参数:
        df: 原始DataFrame
        config: 预处理配置字典

    返回:
        预处理后的DataFrame

    注意: 根据实际需求实现预处理步骤
    """
    logger.info("开始数据预处理")
    df_processed = df.copy()

    # 对传感器缺失的连续NaN做线性插值
    interpolate_cols = ["temperature_c", "relative_humidity", "visibility_km", "wind_speed_ms"]
    existing = [c for c in interpolate_cols if c in df_processed.columns]
    if existing:
        before = df_processed[existing].isnull().sum().sum()
        df_processed[existing] = df_processed[existing].interpolate(method="linear")
        after = df_processed[existing].isnull().sum().sum()
        logger.info(f"线性插值处理 {existing}: 插值前NaN={before}, 插值后NaN={after}")

    logger.info("数据预处理完成")
    return df_processed


# ============================================================================
# 3. 数据判断模块
# ============================================================================

def apply_threshold_judgments(df: pd.DataFrame, judgment_configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    应用阈值判断，大于某值记为1（支持多种判断类型）

    参数:
        df: 原始DataFrame
        judgment_configs: 判断配置列表，每个元素为:
            {
                'column': '列名',           # 要判断的列
                'threshold': 阈值,          # 判断阈值
                'output_column': '输出列名', # 判断结果列名
                'judgment_type': 类型       # 1/2/3，表示不同的判断逻辑
            }

    返回:
        添加了判断列的DataFrame

    注意: 需要根据judgment_type实现不同的判断逻辑
    """
    logger.info(f"开始应用阈值判断，共{len(judgment_configs)}个判断配置")
    df_with_judgments = df.copy()

    for config in judgment_configs:
        column = config['column']
        output_column = config['output_column']
        judgment_type = config.get('judgment_type', 1)
        threshold = config.get('threshold', config.get('thresholds', [0])[0])

        if column not in df.columns:
            logger.warning(f"列 '{column}' 不存在，跳过判断")
            continue

        # TODO: 根据judgment_type实现不同的判断逻辑
        # 示例: 判断类型1 - 大于阈值记为1，否则为0
        if judgment_type == 1:
            df_with_judgments[output_column] = (df[column] > threshold).astype(int)
        # 判断类型2 - 实现其他逻辑
        elif judgment_type == 2:
            # TODO: 实现类型2的判断逻辑
            pass
        # 判断类型3 - 多段分类（支持 thresholds 列表分段）
        # 如 thresholds=[10,20]  →  <=10:0,  10~20:1,  >20:2
        elif judgment_type == 3:
            thresholds = config.get("thresholds", [config.get("threshold", 0)])
            df_with_judgments[output_column] = pd.cut(
                df[column],
                bins=[-float("inf")] + thresholds + [float("inf")],
                labels=range(len(thresholds) + 1),
                right=True
            ).astype(int)
        else:
            logger.warning(f"未知的判断类型: {judgment_type}，使用默认类型1")
            df_with_judgments[output_column] = (df[column] > threshold).astype(int)

        logger.info(f"完成判断: {column} -> {output_column} (类型{judgment_type})")

    return df_with_judgments


# ============================================================================
# 4. 差值计算模块
# ============================================================================

def compute_row_differences(df: pd.DataFrame, diff_columns: List[str], diff_n: int = None) -> pd.DataFrame:
    """
    同时计算两种行间差值：
    1. 相邻行差值（前1行）* 0.1 → 用于加速度
    2. 前 n 行差值（配置文件控制）

    参数:
        df: 原始DataFrame
        diff_columns: 需要计算差值的列名列表
        diff_n: 可选，配置文件中的前n行间隔（不填则只计算相邻行）

    返回:
        添加了两种差值列的DataFrame
    """
    logger.info(f"开始计算行间差值，列: {diff_columns}")
    df_with_diffs = df.copy()

    for column in diff_columns:
        if column not in df.columns:
            logger.warning(f"列 '{column}' 不存在，跳过差值计算")
            continue

        # ======================
        # 1. 原始相邻行差值（前1行）× 0.1 → 加速度
        # ======================
        diff_column_1 = f"{column}_diff"
        df_with_diffs[diff_column_1] = df[column].diff(periods=1) * 0.1
        logger.info(f"完成相邻行差值计算: {column} -> {diff_column_1}")

        # ======================
        # 2. 前 n 行差值（配置控制）
        # ======================
        if diff_n is not None and isinstance(diff_n, int) and diff_n > 0:
            diff_column_n = f"{column}_diff_{diff_n}"
            df_with_diffs[diff_column_n] = df[column].diff(periods=diff_n)
            logger.info(f"完成前{diff_n}行差值计算: {column} -> {diff_column_n}")

    return df_with_diffs


# ============================================================================
# 5. 特征乘积模块
# ============================================================================

def compute_multiply_features(df: pd.DataFrame, multiply_configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    计算两列数值的乘积，生成新特征

    参数:
        df: 原始DataFrame
        multiply_configs: 乘积配置列表，每个元素为:
            {
                'columns': ['列A', '列B'],   # 要相乘的两列
                'output_column': '乘积列名'   # 结果列名
            }

    返回:
        添加了乘积列的DataFrame
    """
    logger.info(f"开始计算特征乘积，共{len(multiply_configs)}个配置")
    df_with_multiply = df.copy()

    for config in multiply_configs:
        cols = config['columns']
        output_column = config['output_column']

        if all(col in df.columns for col in cols):
            df_with_multiply[output_column] = df[cols[0]] * df[cols[1]]
            logger.info(f"完成乘积: {cols[0]} * {cols[1]} -> {output_column}")
        else:
            missing = [col for col in cols if col not in df.columns]
            logger.warning(f"列 {missing} 不存在，跳过乘积计算")

    return df_with_multiply


# ============================================================================
# 6. 滑动窗口统计模块
# ============================================================================

def create_sliding_windows(df: pd.DataFrame, window_sizes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """
    创建滑动窗口索引

    参数:
        df: DataFrame
        window_sizes: 窗口大小列表

    返回:
        窗口索引字典 {window_size: [(start_idx, end_idx), ...]}

    注意: 窗口为左闭右开区间 [start, end)
    """
    logger.info(f"创建滑动窗口，大小: {window_sizes}")
    n_rows = len(df)
    windows = {}

    for window_size in window_sizes:
        if window_size <= 0 or window_size > n_rows:
            logger.warning(f"窗口大小 {window_size} 无效，跳过")
            continue

        window_list = []
        for i in range(n_rows - window_size + 1):
            window_list.append((i, i + window_size))

        windows[window_size] = window_list
        logger.info(f"窗口大小 {window_size}: 创建了 {len(window_list)} 个窗口")

    return windows


def compute_window_statistics(df: pd.DataFrame, window_indices: List[Tuple[int, int]],
                             stat_configs: Dict[str, List[str]], window_size: int) -> pd.DataFrame:
    """
    计算窗口内指定统计量（按 stat_configs 配置）

    参数:
        df: DataFrame
        window_indices: 窗口索引列表 [(start, end), ...]
        stat_configs: 统计配置字典，如 {"mean": ["col1", "col2"], "std": ["col1"]}
        window_size: 窗口大小

    返回:
        包含窗口统计结果的DataFrame，每行对应一个窗口
    """
    logger.info(f"开始计算窗口统计，窗口大小: {window_size}，统计配置: {stat_configs}")

    results = []

    for start, end in window_indices:
        window_data = df.iloc[start:end]
        result_row = {'window_start': start, 'window_end': end - 1, 'window_size': window_size}

        for stat_name, columns in stat_configs.items():
            for column in columns:
                if column not in df.columns:
                    continue

                col_data = window_data[column]

                if stat_name == 'mean':
                    result_row[f"{column}_window{window_size}_mean"] = col_data.mean()
                elif stat_name == 'std':
                    result_row[f"{column}_window{window_size}_std"] = col_data.std()
                elif stat_name == 'min':
                    result_row[f"{column}_window{window_size}_min"] = col_data.min()
                elif stat_name == 'max':
                    result_row[f"{column}_window{window_size}_max"] = col_data.max()
                elif stat_name == 'median':
                    result_row[f"{column}_window{window_size}_median"] = col_data.median()
                else:
                    logger.warning(f"未知的统计类型: {stat_name}，跳过")

        results.append(result_row)

    result_df = pd.DataFrame(results)
    logger.info(f"窗口统计计算完成，生成 {len(result_df)} 行结果")
    return result_df


def compute_window_proportions(df: pd.DataFrame, window_indices: List[Tuple[int, int]],
                              column: str, target_values: List[Any], window_size: int,
                              output_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算窗口内特定值的占比

    参数:
        df: DataFrame
        window_indices: 窗口索引列表
        column: 要统计的列名
        target_values: 要统计的目标值列表
        window_size: 窗口大小
        output_names: 可选，自定义输出列名列表，长度须与 target_values 一致

    返回:
        包含占比统计的DataFrame

    示例: 过去20行中值为1的数据占比
    """
    logger.info(f"开始计算窗口占比，列: {column}，目标值: {target_values}，窗口大小: {window_size}")

    if column not in df.columns:
        logger.warning(f"列 '{column}' 不存在，跳过占比计算")
        return pd.DataFrame()

    results = []

    for start, end in window_indices:
        window_data = df.iloc[start:end]
        col_data = window_data[column]

        result_row = {'window_start': start, 'window_end': end - 1, 'window_size': window_size}

        # 计算每个目标值的占比
        for i, target_value in enumerate(target_values):
            count = (col_data == target_value).sum()
            proportion = count / window_size if window_size > 0 else 0

            # 使用自定义列名或默认命名
            if output_names and i < len(output_names):
                col_name = output_names[i]
            else:
                col_name = f"{column}_value{target_value}_window{window_size}_proportion"
            result_row[col_name] = proportion

        # 计算所有目标值的总占比（仅默认命名时）
        if not output_names:
            total_count = sum((col_data == value).sum() for value in target_values)
            total_proportion = total_count / window_size if window_size > 0 else 0
            result_row[f"{column}_targets_total_window{window_size}_proportion"] = total_proportion

        results.append(result_row)

    result_df = pd.DataFrame(results)
    logger.info(f"窗口占比计算完成，生成 {len(result_df)} 行结果")
    return result_df


# ============================================================================
# 7. 特征整合与列删除模块
# ============================================================================

def combine_features(original_df: pd.DataFrame,
                    diff_features: pd.DataFrame,
                    judgment_features: pd.DataFrame,
                    window_stat_features: pd.DataFrame,
                    window_proportion_features: pd.DataFrame) -> pd.DataFrame:
    """
    整合所有计算的特征

    参数:
        original_df: 原始数据（可能包含预处理后的列）
        diff_features: 差值特征
        judgment_features: 判断特征
        window_stat_features: 窗口统计特征
        window_proportion_features: 窗口占比特征

    返回:
        整合后的特征DataFrame

    注意: 需要根据实际特征结构调整整合逻辑
    """
    logger.info("开始整合所有特征")

    # 复制原始数据作为基础
    combined_df = original_df.copy()

    # TODO: 根据实际特征结构实现合并逻辑
    # 逐步合并各个特征集
    if not judgment_features.empty:
        # 只合并新增的判断列（排除重复的原始列）
        judgment_new_cols = [col for col in judgment_features.columns if col not in combined_df.columns]
        if judgment_new_cols:
            combined_df = pd.concat([combined_df, judgment_features[judgment_new_cols]], axis=1)

    if not diff_features.empty:
        diff_new_cols = [col for col in diff_features.columns if col not in combined_df.columns]
        if diff_new_cols:
            combined_df = pd.concat([combined_df, diff_features[diff_new_cols]], axis=1)

    if not window_stat_features.empty:
        stat_new_cols = [col for col in window_stat_features.columns if col not in combined_df.columns]
        if stat_new_cols:
            combined_df = pd.concat([combined_df, window_stat_features[stat_new_cols]], axis=1)

    if not window_proportion_features.empty:
        prop_new_cols = [col for col in window_proportion_features.columns if col not in combined_df.columns]
        if prop_new_cols:
            combined_df = pd.concat([combined_df, window_proportion_features[prop_new_cols]], axis=1)

    logger.info(f"特征整合完成，最终形状: {combined_df.shape}")
    return combined_df


def drop_specified_columns(df: pd.DataFrame, drop_columns: List[str]) -> pd.DataFrame:
    """
    删除指定的列

    参数:
        df: 原始DataFrame
        drop_columns: 要删除的列名列表

    返回:
        删除指定列后的DataFrame
    """
    if not drop_columns:
        return df

    cols_to_drop = [col for col in drop_columns if col in df.columns]
    if cols_to_drop:
        logger.info(f"删除列: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info("没有需要删除的列")

    return df


# ============================================================================
# 9. 输出模块
# ============================================================================

def save_feature_data(df: pd.DataFrame, output_path: str) -> None:
    """
    保存特征数据到CSV

    参数:
        df: 特征DataFrame
        output_path: 输出文件路径
    """
    try:
        logger.info(f"开始保存特征数据到: {output_path}")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存CSV文件
        df.to_csv(output_path, index=False)
        logger.info(f"特征数据保存成功，行数: {len(df)}，列数: {len(df.columns)}")

    except Exception as e:
        logger.error(f"特征数据保存失败: {e}")
        raise


def generate_statistics_report(feature_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    生成统计报告

    参数:
        feature_df: 特征DataFrame
        output_dir: 输出目录

    返回:
        统计报告字典
    """
    logger.info("开始生成统计报告")

    report = {
        'generation_time': datetime.now().isoformat(),
        'data_shape': feature_df.shape,
        'columns': feature_df.columns.tolist(),
        'data_types': feature_df.dtypes.astype(str).to_dict(),
        'missing_values': feature_df.isnull().sum().to_dict(),
        'basic_statistics': {},
        'summary': {}
    }

    # 数值列的基础统计
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['basic_statistics'] = feature_df[numeric_cols].describe().to_dict()

    # 生成报告文件
    report_file = os.path.join(output_dir, f"feature_statistics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"统计报告保存到: {report_file}")
    except Exception as e:
        logger.error(f"统计报告保存失败: {e}")

    return report


# ============================================================================
# 10. 主函数
# ============================================================================

def main(input_file: str, output_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    主函数，协调整个处理流程

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        config: 配置字典，包含处理参数

    返回:
        处理结果报告字典
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("开始特征计算流程")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"配置参数: {json.dumps(config, indent=2, default=str)}")

    result_report = {
        'start_time': start_time.isoformat(),
        'input_file': input_file,
        'output_file': output_file,
        'config': config,
        'success': False,
        'error_message': None,
        'processing_steps': [],
        'statistics': {}
    }

    try:
        # 步骤1: 加载数据
        logger.info("[步骤1] 加载原始数据")
        raw_df = load_raw_data(input_file)
        result_report['processing_steps'].append('load_raw_data')
        result_report['statistics']['raw_data_shape'] = raw_df.shape

        # 步骤2: 选择处理行数
        logger.info("[步骤2] 选择处理行数")
        n_rows = config.get('n_rows')
        selected_df = select_data_rows(raw_df, n_rows)
        result_report['processing_steps'].append('select_data_rows')
        result_report['statistics']['selected_data_shape'] = selected_df.shape

        # 步骤3: 数据质量验证
        logger.info("[步骤3] 数据质量验证")
        quality_report = validate_data_quality(selected_df)
        result_report['processing_steps'].append('validate_data_quality')
        result_report['quality_report'] = quality_report

        # 步骤4: 数据预处理
        logger.info("[步骤4] 数据预处理")
        preprocess_config = config.get('preprocess_config', {})
        processed_df = preprocess_data(selected_df, preprocess_config)
        result_report['processing_steps'].append('preprocess_data')

        # 步骤5: 应用阈值判断
        logger.info("[步骤5] 应用阈值判断")
        judgment_configs = config.get('judgment_configs', [])
        if judgment_configs:
            judgment_df = apply_threshold_judgments(processed_df, judgment_configs)
            result_report['processing_steps'].append('apply_threshold_judgments')
        else:
            judgment_df = processed_df.copy()
            logger.info("无阈值判断配置，跳过此步骤")

        # 步骤6: 计算行间差值
        logger.info("[步骤6] 计算行间差值")
        diff_columns = config.get('diff_columns', [])
        if diff_columns:
            diff_df = compute_row_differences(judgment_df, diff_columns)
            result_report['processing_steps'].append('compute_row_differences')
        else:
            diff_df = judgment_df.copy()
            logger.info("无差值计算配置，跳过此步骤")

        # 步骤7: 计算特征乘积
        logger.info("[步骤7] 计算特征乘积")
        multiply_configs = config.get('multiply_features', [])
        if multiply_configs:
            multiply_df = compute_multiply_features(diff_df, multiply_configs)
            result_report['processing_steps'].append('compute_multiply_features')
        else:
            multiply_df = diff_df.copy()
            logger.info("无特征乘积配置，跳过此步骤")

        # 步骤8: 滑动窗口统计
        logger.info("[步骤8] 滑动窗口统计")
        window_sizes = config.get('window_sizes', [])
        stat_configs = config.get('stat_configs', {})

        window_stat_results = pd.DataFrame()
        if window_sizes and stat_configs:
            windows = create_sliding_windows(multiply_df, window_sizes)

            window_stats_list = []
            for window_size, window_indices in windows.items():
                stats_df = compute_window_statistics(multiply_df, window_indices, stat_configs, window_size)
                window_stats_list.append(stats_df)

            if window_stats_list:
                window_stat_results = pd.concat(window_stats_list, axis=1)
                result_report['processing_steps'].append('compute_window_statistics')
        else:
            logger.info("无窗口统计配置，跳过此步骤")

        # 步骤9: 窗口占比计算
        logger.info("[步骤9] 窗口占比计算")
        proportion_configs = config.get('proportion_configs', [])

        window_proportion_results = pd.DataFrame()
        if window_sizes and proportion_configs:
            if 'windows' not in locals():
                windows = create_sliding_windows(multiply_df, window_sizes)

            proportion_stats_list = []
            for config_item in proportion_configs:
                column = config_item.get('column')
                target_values = config_item.get('target_values', [])
                window_size = config_item.get('window_size')
                output_names = config_item.get('output_names')

                if column and target_values and window_size and window_size in windows:
                    proportions_df = compute_window_proportions(
                        multiply_df, windows[window_size],
                        column, target_values, window_size,
                        output_names
                    )
                    proportion_stats_list.append(proportions_df)

            if proportion_stats_list:
                window_proportion_results = pd.concat(proportion_stats_list, axis=1)
                result_report['processing_steps'].append('compute_window_proportions')
        else:
            logger.info("无窗口占比配置，跳过此步骤")

        # 步骤10: 整合特征
        logger.info("[步骤10] 整合所有特征")
        final_features = combine_features(
            multiply_df,
            multiply_df,
            judgment_df if judgment_configs else pd.DataFrame(),
            window_stat_results,
            window_proportion_results
        )
        result_report['processing_steps'].append('combine_features')
        result_report['statistics']['final_features_shape'] = final_features.shape

        # 步骤11: 删除指定列
        logger.info("[步骤11] 删除指定列")
        drop_columns = config.get('drop_columns', [])
        if drop_columns:
            final_features = drop_specified_columns(final_features, drop_columns)
            result_report['processing_steps'].append('drop_columns')
        else:
            logger.info("无列删除配置，跳过此步骤")

        # 步骤12: 保存特征数据
        logger.info("[步骤12] 保存特征数据")
        save_feature_data(final_features, output_file)
        result_report['processing_steps'].append('save_feature_data')

        # 步骤13: 生成统计报告
        logger.info("[步骤13] 生成统计报告")
        output_dir = os.path.dirname(output_file)
        stats_report = generate_statistics_report(final_features, output_dir)
        result_report['statistics_report'] = stats_report
        result_report['processing_steps'].append('generate_statistics_report')

        # 更新成功状态
        result_report['success'] = True
        result_report['end_time'] = datetime.now().isoformat()
        processing_time = (datetime.now() - start_time).total_seconds()
        result_report['processing_time_seconds'] = processing_time

        logger.info(f"特征计算流程完成，总耗时: {processing_time:.2f}秒")
        logger.info(f"最终特征数据形状: {final_features.shape}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"特征计算流程失败: {e}")
        result_report['success'] = False
        result_report['error_message'] = str(e)
        result_report['end_time'] = datetime.now().isoformat()

    return result_report


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    """
    命令行使用示例:
    python compute_feature.py --input data/aligned/aligned_data_refined_soc.csv \\
                              --output data/aligned/feature_data.csv \\
                              --config configs/feature_config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description='计算原始数据的统计特征')
    parser.add_argument('--input', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出CSV文件路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径（JSON格式）')
    parser.add_argument('--n_rows', type=int,
                       help='处理的行数限制（可选）')

    args = parser.parse_args()

    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"从文件加载配置: {args.config}")
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")

    # 命令行参数覆盖配置文件
    if args.n_rows is not None:
        config['n_rows'] = args.n_rows

    # 运行主函数
    result = main(args.input, args.output, config)

    # 输出结果摘要
    if result['success']:
        print(f"\n处理成功!")
        print(f"输入文件: {result['input_file']}")
        print(f"输出文件: {result['output_file']}")
        print(f"处理行数: {result['statistics'].get('selected_data_shape', (0, 0))[0]}")
        print(f"特征列数: {result['statistics'].get('final_features_shape', (0, 0))[1]}")
        print(f"处理时间: {result.get('processing_time_seconds', 0):.2f}秒")
    else:
        print(f"\n处理失败!")
        print(f"错误信息: {result.get('error_message', '未知错误')}")
        exit(1)
