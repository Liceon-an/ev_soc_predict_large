#!/usr/bin/env python3
"""
测试特征工程模块
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.FeatureEngineering.feature_engineer import FeatureEngineer, create_feature_engineering_config

def create_test_data(n_samples=100):
    """创建测试数据"""
    np.random.seed(42)
    
    # 创建时间序列
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='10S')
    
    # 创建测试数据
    data = {
        'DATE': timestamps,
        'speed': np.random.uniform(0, 100, n_samples),  # 0-100 km/h
        'mileage': np.cumsum(np.random.uniform(0, 0.1, n_samples)),  # 累计里程
        'total_volt': np.random.uniform(350, 400, n_samples),  # 电压
        'total_current': np.random.uniform(-50, 100, n_samples),  # 电流（可正可负）
        'max_cell_volt': np.random.uniform(3.8, 4.0, n_samples),
        'min_cell_volt': np.random.uniform(3.7, 3.9, n_samples),
        'max_temp': np.random.uniform(20, 40, n_samples),
        'min_temp': np.random.uniform(15, 35, n_samples),
        'temperature_c': np.random.uniform(15, 35, n_samples),
        'relative_humidity': np.random.uniform(30, 80, n_samples),
        'visibility_km': np.random.uniform(5, 20, n_samples),
        'wind_speed_ms': np.random.uniform(0, 10, n_samples),
        'time_diff': np.full(n_samples, 10.0),  # 固定10秒间隔
        'refined_soc': np.random.uniform(20, 90, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def test_basic_features():
    """测试基础特征计算"""
    print("测试基础特征计算")
    print("=" * 60)
    
    # 创建测试数据
    test_df = create_test_data(50)
    print(f"测试数据形状: {test_df.shape}")
    print(f"列名: {list(test_df.columns)}")
    
    # 创建特征工程配置
    config = create_feature_engineering_config(enabled=True, use_speed_clustering=True)
    
    # 创建特征工程对象
    engineer = FeatureEngineer(config)
    
    # 拟合模型（计算速度阈值）
    engineer.fit(test_df)
    
    # 计算特征（单点模式）
    print("\n1. 测试单点特征计算:")
    single_point_df = test_df.iloc[:10].copy()
    enhanced_df = engineer.compute_features(single_point_df, is_sequence=False)
    
    print(f"原始数据列数: {len(single_point_df.columns)}")
    print(f"增强数据列数: {len(enhanced_df.columns)}")
    print(f"新增列: {set(enhanced_df.columns) - set(single_point_df.columns)}")
    
    # 检查具体特征
    print("\n2. 检查具体特征值:")
    if 'acceleration' in enhanced_df.columns:
        print(f"加速度范围: [{enhanced_df['acceleration'].min():.2f}, {enhanced_df['acceleration'].max():.2f}]")
    
    if 'speed_category' in enhanced_df.columns:
        categories = enhanced_df['speed_category'].unique()
        print(f"速度类别: {sorted(categories)}")
    
    if 'instant_power_kw' in enhanced_df.columns:
        print(f"瞬时功率范围: [{enhanced_df['instant_power_kw'].min():.2f}, {enhanced_df['instant_power_kw'].max():.2f}] kW")
    
    if 'cruising_ratio' in enhanced_df.columns:
        print(f"巡航时间比: {enhanced_df['cruising_ratio'].iloc[0]:.2%}")
    
    # 测试序列模式
    print("\n3. 测试序列特征计算:")
    sequence_df = test_df.iloc[:20].copy()
    enhanced_sequence_df = engineer.compute_features(sequence_df, is_sequence=True)
    
    if 'segment_distance' in enhanced_sequence_df.columns:
        print(f"片段里程: {enhanced_sequence_df['segment_distance'].iloc[0]:.4f} km")
    
    if 'total_energy_kwh' in enhanced_sequence_df.columns:
        print(f"片段总能耗: {enhanced_sequence_df['total_energy_kwh'].iloc[0]:.6f} kWh")
    
    # 获取特征名称
    original_features = ['speed', 'mileage', 'total_volt', 'total_current']
    all_features = engineer.get_feature_names(original_features)
    print(f"\n4. 特征名称列表 ({len(all_features)}个特征):")
    for i, feature in enumerate(all_features[:15], 1):
        print(f"  {i:2d}. {feature}")
    if len(all_features) > 15:
        print(f"  ... 还有 {len(all_features) - 15} 个特征")
    
    print("\n" + "=" * 60)
    print("基础特征测试完成!")
    
    return enhanced_df

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况")
    print("=" * 60)
    
    # 创建边界测试数据
    edge_data = {
        'speed': [0, 0, 0, 0, 0],  # 全零速度
        'mileage': [100, 100, 100, 100, 100],  # 里程不变
        'total_volt': [380, 380, 380, 380, 380],
        'total_current': [0, 0, 0, 0, 0],  # 零电流
        'time_diff': [0, 10, 10, 10, 10],  # 第一个时间间隔为0
        'refined_soc': [50, 50, 50, 50, 50]
    }
    
    edge_df = pd.DataFrame(edge_data)
    
    # 创建特征工程配置
    config = create_feature_engineering_config(enabled=True)
    engineer = FeatureEngineer(config)
    engineer.fit(edge_df)
    
    # 计算特征
    enhanced_edge_df = engineer.compute_features(edge_df, is_sequence=True)
    
    print(f"边界数据形状: {edge_df.shape} -> {enhanced_edge_df.shape}")
    
    # 检查处理结果
    if 'acceleration' in enhanced_edge_df.columns:
        print(f"加速度（含零时间间隔）: {enhanced_edge_df['acceleration'].tolist()}")
    
    if 'cruising_ratio' in enhanced_edge_df.columns:
        print(f"巡航时间比（全零速度）: {enhanced_edge_df['cruising_ratio'].iloc[0]:.2%}")
    
    if 'instant_power_kw' in enhanced_edge_df.columns:
        print(f"瞬时功率（零电流）: {enhanced_edge_df['instant_power_kw'].tolist()}")
    
    print("\n" + "=" * 60)
    print("边界情况测试完成!")

def test_config_serialization():
    """测试配置序列化"""
    print("\n测试配置序列化")
    print("=" * 60)
    
    # 创建配置
    config = create_feature_engineering_config(
        enabled=True,
        use_speed_clustering=True,
        n_clusters=4,
        consider_charging=True
    )
    
    # 创建特征工程对象
    engineer = FeatureEngineer(config)
    
    # 获取配置
    saved_config = engineer.get_config()
    
    print("原始配置:")
    print(f"  启用: {config['feature_engineering']['enabled']}")
    print(f"  聚类数: {config['feature_engineering']['speed_clustering']['n_clusters']}")
    print(f"  考虑充电: {config['feature_engineering']['power']['consider_charging']}")
    
    print("\n保存的配置:")
    print(f"  启用: {saved_config['enabled']}")
    print(f"  聚类数: {saved_config['speed_clustering']['n_clusters']}")
    print(f"  考虑充电: {saved_config['power']['consider_charging']}")
    
    # 验证配置一致性
    assert saved_config['enabled'] == config['feature_engineering']['enabled']
    assert saved_config['speed_clustering']['n_clusters'] == config['feature_engineering']['speed_clustering']['n_clusters']
    assert saved_config['power']['consider_charging'] == config['feature_engineering']['power']['consider_charging']
    
    print("\n✅ 配置序列化测试通过!")
    print("=" * 60)

def main():
    """主测试函数"""
    print("特征工程模块测试套件")
    print("=" * 60)
    
    try:
        # 测试基础特征
        enhanced_df = test_basic_features()
        
        # 测试边界情况
        test_edge_cases()
        
        # 测试配置序列化
        test_config_serialization()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        
        # 显示示例数据
        print("\n示例增强数据（前5行）:")
        print(enhanced_df.head().to_string())
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
