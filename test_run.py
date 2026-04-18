#!/usr/bin/env python3
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f当前目录: {current_dir})
print(fPython路径: {sys.path})

try:
    from src.DataProcessing.enhanced_preprocessor import main
    print(导入成功!)
    
    # 模拟命令行参数
    sys.argv = [
        'enhanced_preprocessor.py',
        '--config', 'configs/config_enhanced.yaml',
        '--input', 'data/aligned/aligned_data_refined_soc.csv',
        '--output', 'data/processed/enhanced_test',
        '--log-level', 'INFO'
    ]
    
    print(开始运行主函数...)
    main()
    
except Exception as e:
    print(f错误: {e})
    import traceback
    traceback.print_exc()
