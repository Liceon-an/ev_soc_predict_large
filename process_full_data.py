#!/usr/bin/env python3
"""
处理全部数据的脚本
用于运行增强预处理器处理完整数据集
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_logging():
    """设置日志配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"full_data_processing_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def process_full_data():
    """处理完整数据集"""
    logger = setup_logging()

    try:
        logger.info("=" * 60)
        logger.info("开始处理完整数据集")
        logger.info("=" * 60)

        # 导入必要的模块
        import yaml
        import pandas as pd
        from src.DataProcessing.enhanced_preprocessor import EnhancedDataPreparator, load_config

        # 1. 加载配置
        config_path = "configs/config_enhanced.yaml"
        logger.info(f"加载配置文件: {config_path}")
        config = load_config(config_path)

        # 修改输入文件路径为完整数据
        config['data_processing']['input_file'] = 'data/aligned/aligned_data_refined_soc.csv'

        # 2. 创建预处理器
        logger.info("创建增强预处理器...")
        preparator = EnhancedDataPreparator(config)

        # 3. 加载完整数据
        input_file = 'data/aligned/aligned_data_refined_soc.csv'
        logger.info(f"加载完整数据: {input_file}")

        start_time = time.time()
        df = preparator.load_data(input_file)
        load_time = time.time() - start_time
        logger.info(f"数据加载完成，耗时: {load_time:.2f}秒")
        logger.info(f"数据形状: {df.shape}")

        # 4. 处理数据
        logger.info("开始数据处理流程...")
        process_start = time.time()

        X, y, metadata = preparator.process(df)

        process_time = time.time() - process_start
        logger.info(f"数据处理完成，耗时: {process_time:.2f}秒")
        logger.info(f"生成样本数: {len(X)}")
        logger.info(f"特征维度: {X.shape[1]} × {X.shape[2]}")

        # 5. 保存数据
        output_dir = 'data/processed/enhanced_full'
        logger.info(f"保存数据到: {output_dir}")

        save_start = time.time()
        saved_files = preparator.save_processed_data(X, y, metadata, output_dir)
        save_time = time.time() - save_start

        logger.info(f"数据保存完成，耗时: {save_time:.2f}秒")

        # 6. 统计信息
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("处理完成统计:")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  数据加载: {load_time:.2f}秒 ({load_time/total_time*100:.1f}%)")
        logger.info(f"  数据处理: {process_time:.2f}秒 ({process_time/total_time*100:.1f}%)")
        logger.info(f"  数据保存: {save_time:.2f}秒 ({save_time/total_time*100:.1f}%)")
        logger.info(f"  生成样本: {len(X)}个")
        logger.info(f"  特征维度: {X.shape[2]}维")
        logger.info(f"  输出目录: {output_dir}")

        # 7. 详细统计信息
        logger.info("\n详细统计信息:")
        logger.info(f"  原始数据行数: {len(df)}")
        logger.info(f"  有效序列数: {len(X)}")
        logger.info(f"  序列长度: {X.shape[1]} × {X.shape[2]}")
        logger.info(f"  标签范围: {y.min():.1f}% - {y.max():.1f}%")
        logger.info(f"  标签均值: {y.mean():.2f}%")

        # 8. 检查数据质量
        logger.info("\n数据质量检查:")
        import numpy as np
        logger.info(f"  X中NaN比例: {np.isnan(X).mean():.2%}")
        logger.info(f"  X中Inf比例: {np.isinf(X).mean():.2%}")
        logger.info(f"  y中NaN比例: {np.isnan(y).mean():.2%}")

        # 9. 保存处理报告
        report_file = os.path.join(output_dir, "processing_report.txt")
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("完整数据处理报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {config_path}\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"输出目录: {output_dir}\n\n")
            f.write(f"原始数据行数: {len(df)}\n")
            f.write(f"生成样本数: {len(X)}\n")
            f.write(f"特征维度: {X.shape[2]}\n")
            f.write(f"序列长度: {X.shape[1]}\n")
            f.write(f"总耗时: {total_time:.2f}秒\n")
            f.write(f"数据加载: {load_time:.2f}秒\n")
            f.write(f"数据处理: {process_time:.2f}秒\n")
            f.write(f"数据保存: {save_time:.2f}秒\n\n")
            f.write("输出文件:\n")
            for key, path in saved_files.items():
                if path:
                    f.write(f"  {key}: {path}\n")

        logger.info(f"处理报告已保存: {report_file}")
        logger.info("=" * 60)
        logger.info("完整数据处理完成!")

        return True

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_bash_script():
    """创建bash运行脚本"""
    script_content = '''#!/bin/bash
# 处理完整数据的bash脚本

echo "============================================================"
echo "开始处理完整数据集"
echo "============================================================"

# 激活conda环境
source /root/miniconda3/bin/activate

# 进入项目目录
cd /root/code/ev_soc_predict

# 运行Python脚本
echo "运行处理脚本..."
python process_full_data.py

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "============================================================"
    echo "处理成功完成!"
    echo "============================================================"
else
    echo "============================================================"
    echo "处理失败!"
    echo "============================================================"
    exit 1
fi

# 显示输出文件
echo ""
echo "输出文件:"
find data/processed/enhanced_full -type f -name "*.npz" -o -name "*.yaml" -o -name "*.txt" | sort
echo ""
echo "日志文件:"
find logs -name "full_data_processing_*.log" | sort | tail -1
'''

    script_path = "process_full_data.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # 添加执行权限
    import stat
    os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

    return script_path

def create_optimized_config():
    """创建优化后的配置文件"""
    optimized_config = {
        'experiment_name': 'lstm_enhanced_full',
        'processing_mode': 'enhanced',

        'data_processing': {
            'input_file': 'data/aligned/aligned_data_refined_soc.csv',
            'sequence_length': 40,
            'stride': 20,
            'max_time_gap': 3600,  # 调整为1小时，适应数据特性
            'time_col': 'DATE',
            'label_col': 'refined_soc',
            'feature_cols': [
                'speed', 'mileage', 'total_volt', 'total_current',
                'max_cell_volt', 'min_cell_volt', 'max_temp', 'min_temp',
                'temperature_c', 'relative_humidity', 'visibility_km', 'wind_speed_ms'
            ],
            'split': {'validation': 0.15, 'test': 0.15}
        },

        'enhanced_processing': {
            'enabled': True,
            'context_size': 2,
            'min_segment_length': 44,
            'continuity_detection': True
        },

        'feature_engineering': {
            'enabled': True,
            'base_features': [
                'speed', 'total_volt', 'total_current', 'refined_soc', 'mileage',
                'max_temp', 'min_temp', 'max_cell_volt', 'min_cell_volt',
                'temperature_c', 'relative_humidity', 'visibility_km', 'wind_speed_ms'
            ],
            'feature_types': {
                'point_features': True,
                'sequence_features': True,
                'boundary_features': True
            },
            'speed_clustering': {
                'enabled': False,  # 禁用K-means，使用分位数
                'use_quantiles': True,
                'quantiles': [0.33, 0.67],
                'n_clusters': 3
            },
            'acceleration': {
                'min_time_diff': 0.1,
                'max_acceleration': 10.0
            },
            'power': {
                'consider_charging': False
            }
        },

        'model': {
            'input_size': None,  # 运行时自动计算
            'hidden_size': 128,
            'num_layers': 3,
            'output_size': 1
        },

        'training': {
            'optimizer': 'adam',
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 108,
            'patience': 30,
            'loss_function': 'mae',
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 5
            }
        },

        'paths': {
            'processed_data_file': 'lstm_enhanced_full_data.npz',
            'processed_scaler_file': 'lstm_enhanced_full_scalers.pkl',
            'model_save_name': 'best_lstm_enhanced_full_model.pth',
            'plot_save_name': 'lstm_enhanced_full_prediction_scatter.png',
            'csv_save_name': 'lstm_enhanced_full_predictions_detail.csv',
            'log_dir': 'logs'
        }
    }

    import yaml
    config_path = "configs/config_enhanced_optimized.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)

    return config_path

if __name__ == "__main__":
    print("准备处理完整数据...")

    # 创建优化配置文件
    print("1. 创建优化配置文件...")
    optimized_config = create_optimized_config()
    print(f"   优化配置文件已创建: {optimized_config}")

    # 创建bash脚本
    print("2. 创建bash运行脚本...")
    bash_script = create_bash_script()
    print(f"   Bash脚本已创建: {bash_script}")

    print("\n3. 运行处理脚本...")
    print("=" * 60)

    success = process_full_data()

    if success:
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 检查输出文件: data/processed/enhanced_full/")
        print("2. 查看日志文件: logs/full_data_processing_*.log")
        print("3. 验证数据质量")
        print("4. 进行模型训练测试")
    else:
        print("\n" + "=" * 60)
        print("处理失败!")
        print("=" * 60)
        sys.exit(1)