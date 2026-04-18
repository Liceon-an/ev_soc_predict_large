#!/usr/bin/env python3
"""
时间片段消融实验配置文件生成脚本
生成不同时间窗口长度的配置文件：200s, 400s, 600s, 800s, 1000s
"""

import yaml
import os
from pathlib import Path

def generate_time_window_configs(base_config_path, window_lengths, time_resolution=10):
    """
    生成不同时间窗口长度的配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        window_lengths: 时间窗口长度列表（秒）
        time_resolution: 时间分辨率（秒/步）
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    config_dir = Path(base_config_path).parent
    
    for window_seconds in window_lengths:
        # 计算序列长度（时间步数）
        seq_len = window_seconds // time_resolution
        
        # 创建新配置（深拷贝）
        new_config = base_config.copy()
        
        # 更新实验名称
        new_config['experiment_name'] = f"lstm_{window_seconds}s"
        
        # 更新数据预处理参数
        new_config['data_processing']['sequence_length'] = seq_len
        new_config['data_processing']['stride'] = max(seq_len // 2, 1)  # 50%重叠，至少为1
        
        # 根据序列长度调整最大时间间隙
        # 时间间隙应至少为序列长度的2倍
        current_max_gap = new_config['data_processing'].get('max_time_gap', 500)
        if current_max_gap < seq_len * 2:
            new_config['data_processing']['max_time_gap'] = seq_len * 2
        
        # 更新文件路径
        new_config['paths']['processed_data_file'] = f"lstm_{window_seconds}s_data.npz"
        new_config['paths']['processed_scaler_file'] = f"lstm_{window_seconds}s_scalers.pkl"
        new_config['paths']['model_save_name'] = f"best_lstm_{window_seconds}s_model.pth"
        new_config['paths']['plot_save_name'] = f"lstm_{window_seconds}s_prediction_scatter.png"
        new_config['paths']['csv_save_name'] = f"lstm_{window_seconds}s_predictions_detail.csv"
        
        # 保存配置文件
        output_path = config_dir / f"config_{window_seconds}s.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 生成配置文件: {output_path}")
        print(f"   序列长度: {seq_len} 步 ({window_seconds}秒)")
        print(f"   步长: {new_config['data_processing']['stride']} 步")
        print(f"   最大时间间隙: {new_config['data_processing']['max_time_gap']} 秒")

def main():
    """主函数"""
    # 项目根目录
    project_root = Path("/root/code/ev_soc_predict")
    base_config_path = project_root / "configs" / "config_400s.yaml"
    
    # 检查基础配置文件是否存在
    if not base_config_path.exists():
        print(f"❌ 基础配置文件不存在: {base_config_path}")
        return
    
    # 要生成的时间窗口长度（秒）
    window_lengths = [200, 400, 600, 800, 1000]
    
    print("=" * 60)
    print("时间片段消融实验配置文件生成器")
    print("=" * 60)
    print(f"基础配置: {base_config_path}")
    print(f"时间窗口: {window_lengths}")
    print(f"时间分辨率: 10秒/步")
    print("=" * 60)
    
    # 生成配置文件
    generate_time_window_configs(base_config_path, window_lengths)
    
    print("=" * 60)
    print("✅ 所有配置文件生成完成！")
    print("=" * 60)
    
    # 显示生成的配置文件列表
    config_dir = project_root / "configs"
    print("\n生成的配置文件:")
    for window in window_lengths:
        config_file = config_dir / f"config_{window}s.yaml"
        if config_file.exists():
            print(f"  - {config_file.name}")

if __name__ == "__main__":
    main()
