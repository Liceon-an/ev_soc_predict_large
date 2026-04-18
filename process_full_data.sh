#!/bin/bash
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
