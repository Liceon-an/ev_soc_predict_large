#!/bin/bash
# 处理完整数据的简单脚本

echo "============================================================"
echo "ev_soc_predict 完整数据处理脚本"
echo "开始时间: $(date)"
echo "============================================================"

# 设置环境
echo "1. 设置环境..."
source /root/miniconda3/bin/activate
cd /root/code/ev_soc_predict

# 检查文件
echo "2. 检查文件..."
if [ ! -f "data/aligned/aligned_data_refined_soc.csv" ]; then
    echo "错误: 数据文件不存在!"
    exit 1
fi

if [ ! -f "configs/config_enhanced.yaml" ]; then
    echo "错误: 配置文件不存在!"
    exit 1
fi

echo "  数据文件: data/aligned/aligned_data_refined_soc.csv"
echo "  配置文件: configs/config_enhanced.yaml"

# 创建输出目录
echo "3. 准备输出目录..."
mkdir -p data/processed/enhanced_full
mkdir -p logs

# 运行处理
echo "4. 开始数据处理..."
echo "   这可能需要一些时间，请耐心等待..."
echo "   查看实时日志: tail -f logs/full_processing_$(date +%Y%m%d_%H%M%S).log"
echo ""

# 运行Python处理脚本
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_processing_${TIMESTAMP}.log"

python -c "
import sys
import os
import time
import logging
from datetime import datetime

# 设置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = '${LOG_FILE}'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    logger.info('=' * 60)
    logger.info('开始处理完整数据集')
    logger.info('=' * 60)

    import yaml
    import pandas as pd
    import numpy as np

    # 添加项目路径
    sys.path.insert(0, '.')
    from src.DataProcessing.enhanced_preprocessor import EnhancedDataPreparator

    # 加载配置
    config_path = 'configs/config_enhanced.yaml'
    logger.info(f'加载配置文件: {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 修改输入文件
    config['data_processing']['input_file'] = 'data/aligned/aligned_data_refined_soc.csv'

    # 创建预处理器
    preparator = EnhancedDataPreparator(config)

    # 加载数据
    logger.info('加载数据...')
    start_time = time.time()
    df = pd.read_csv('data/aligned/aligned_data_refined_soc.csv')
    load_time = time.time() - start_time
    logger.info(f'数据加载完成: {df.shape}, 耗时: {load_time:.2f}秒')

    # 处理数据
    logger.info('开始数据处理...')
    process_start = time.time()
    X, y, metadata = preparator.process(df)
    process_time = time.time() - process_start
    logger.info(f'数据处理完成: X={X.shape}, y={y.shape}, 耗时: {process_time:.2f}秒')

    # 保存数据
    output_dir = 'data/processed/enhanced_full'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'lstm_enhanced_full_{timestamp}'

    # 保存数据文件
    data_file = os.path.join(output_dir, f'{base_name}_data.npz')
    np.savez_compressed(data_file, X=X, y=y)

    # 保存元数据
    metadata_file = os.path.join(output_dir, f'{base_name}_metadata.yaml')
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    # 保存特征名称
    if 'feature_names' in metadata:
        feature_file = os.path.join(output_dir, f'{base_name}_features.txt')
        with open(feature_file, 'w') as f:
            for feature in metadata['feature_names']:
                f.write(f'{feature}\\n')

    total_time = time.time() - start_time

    logger.info('=' * 60)
    logger.info('处理完成!')
    logger.info(f'总耗时: {total_time:.2f}秒')
    logger.info(f'原始数据: {df.shape}')
    logger.info(f'生成样本: {X.shape}')
    logger.info(f'输出目录: {output_dir}')
    logger.info('=' * 60)

    # 打印输出文件
    print('\\n输出文件:')
    print(f'  数据文件: {data_file}')
    print(f'  元数据: {metadata_file}')
    if 'feature_names' in metadata:
        print(f'  特征文件: {feature_file}')

except Exception as e:
    logger.error(f'处理失败: {e}')
    import traceback
    traceback.print_exc()
    raise
" 2>&1 | tee -a "${LOG_FILE}"

# 检查处理结果
echo ""
echo "============================================================"
echo "处理完成!"
echo "结束时间: $(date)"
echo "============================================================"

# 显示输出文件
echo ""
echo "输出文件列表:"
find data/processed/enhanced_full -type f -name "*.npz" -o -name "*.yaml" -o -name "*.txt" 2>/dev/null | sort

# 显示最新文件
echo ""
echo "最新输出文件:"
ls -la data/processed/enhanced_full/*.npz 2>/dev/null | tail -1
ls -la data/processed/enhanced_full/*.yaml 2>/dev/null | tail -1

echo ""
echo "日志文件: ${LOG_FILE}"
echo "可以使用以下命令查看日志:"
echo "  tail -n 50 ${LOG_FILE}"
echo "  cat ${LOG_FILE} | grep -i 'error\|warning\|info'"

echo ""
echo "============================================================"
echo "下一步建议:"
echo "1. 检查数据质量: python -c \"import numpy as np; data=np.load('data/processed/enhanced_full/latest.npz'); print(f'X形状: {data[\"X\"].shape}, y形状: {data[\"y\"].shape}')\""
echo "2. 验证特征计算"
echo "3. 进行模型训练测试"
echo "============================================================"