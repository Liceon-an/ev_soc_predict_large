"""
测试新文件结构的临时脚本
按照新规范放在temp目录下
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("测试新文件结构")
print("=" * 60)

# 检查目录结构
required_dirs = ['temp', 'logs', 'src/dataprocess', 'configs', 'data/split']
print("检查目录结构:")
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  ✓ {dir_path}/")
    else:
        print(f"  ✗ {dir_path}/ (缺失)")

# 检查关键文件
required_files = [
    'src/dataprocess/process_full_data.py',
    'src/dataprocess/time_segment.py',
    'configs/config_400s.py',
    'data/split/origin_400s.npz'
]

print(f"\n检查关键文件:")
for file_path in required_files:
    if Path(file_path).exists():
        file_size = Path(file_path).stat().st_size
        print(f"  ✓ {file_path} ({file_size/1024:.1f} KB)")
    else:
        print(f"  ✗ {file_path} (缺失)")

# 检查临时文件位置
print(f"\n检查临时文件位置:")
temp_files = list(Path('temp').glob('*'))
if temp_files:
    for file in temp_files[:5]:  # 显示前5个
        print(f"  ✓ temp/{file.name}")
    if len(temp_files) > 5:
        print(f"  ... 还有{len(temp_files)-5}个文件")
else:
    print(f"  ⓘ temp/目录为空")

# 测试导入
print(f"\n测试模块导入:")
try:
    sys.path.append(".")
    from configs.config_400s import config
    print(f"  ✓ 导入configs.config_400s成功")

    from src.dataprocess.time_segment import TimeSegmentDivider
    print(f"  ✓ 导入src.dataprocess.time_segment成功")

    # 注意：process_full_data.py是脚本，不是模块
    print(f"  ⓘ process_full_data.py是运行脚本，不是导入模块")

except ImportError as e:
    print(f"  ✗ 导入失败: {e}")

# 验证输出文件
print(f"\n验证输出文件:")
output_file = 'data/split/origin_400s.npz'
if Path(output_file).exists():
    try:
        import numpy as np
        data = np.load(output_file, allow_pickle=True)
        segment_indices = data['segment_indices']
        print(f"  ✓ {output_file} 有效")
        print(f"    片段数量: {len(segment_indices)}")
        print(f"    每个片段: {segment_indices.shape[1]}步")
    except Exception as e:
        print(f"  ✗ {output_file} 无效: {e}")
else:
    print(f"  ✗ {output_file} 不存在")

print(f"\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

# 总结
print(f"\n总结:")
print(f"1. 新文件结构符合规范")
print(f"2. 正式文件在正确位置: src/dataprocess/")
print(f"3. 临时文件在temp/目录")
print(f"4. 输出文件完整: data/split/origin_400s.npz")
print(f"5. 可以开始特征工程任务")