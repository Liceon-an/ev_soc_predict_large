#!/usr/bin/env python3
"""
测试消融实验框架
"""

import subprocess
import time
from pathlib import Path

def test_single_experiment(config_path):
    """测试单个实验"""
    config_path = Path(config_path)
    config_name = config_path.stem
    
    print(f"\n测试实验: {config_name}")
    print("="*40)
    
    # 1. 测试数据预处理
    print("1. 测试数据预处理...")
    preprocess_cmd = [
        "/root/miniconda3/bin/python",
        "src/DataProcessing/preprocess.py",
        "--config", str(config_path)
    ]
    
    try:
        result = subprocess.run(
            preprocess_cmd,
            cwd="/root/code/ev_soc_predict",
            capture_output=True,
            text=True,
            timeout=600  # 10分钟
        )
        
        if result.returncode == 0:
            print("   ✅ 数据预处理成功")
            
            # 检查输出文件
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            data_file = Path("/root/code/ev_soc_predict/data/processed") / config['paths']['processed_data_file']
            scaler_file = Path("/root/code/ev_soc_predict/data/processed") / config['paths']['processed_scaler_file']
            
            if data_file.exists() and scaler_file.exists():
                print(f"   ✅ 数据文件存在: {data_file.name}")
                print(f"   ✅ Scaler文件存在: {scaler_file.name}")
            else:
                print(f"   ❌ 输出文件缺失")
                return False
        else:
            print(f"   ❌ 数据预处理失败: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ 数据预处理超时")
        return False
    
    # 2. 测试训练（简化版，只运行几个epoch）
    print("2. 测试训练脚本...")
    
    # 先备份原始配置
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建测试配置（减少epoch数）
    test_config = config.copy()
    test_config['training']['epochs'] = 3  # 只训练3个epoch
    test_config['training']['patience'] = 2  # 早停耐心值
    
    test_config_path = config_path.parent / f"test_{config_path.name}"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    train_cmd = [
        "/root/miniconda3/bin/python",
        "src/lstm.py",
        "--config", str(test_config_path)
    ]
    
    try:
        result = subprocess.run(
            train_cmd,
            cwd="/root/code/ev_soc_predict",
            capture_output=True,
            text=True,
            timeout=300  # 5分钟
        )
        
        # 删除测试配置
        test_config_path.unlink(missing_ok=True)
        
        if result.returncode == 0:
            print("   ✅ 训练脚本成功")
            
            # 检查输出中是否有关键指标
            if 'Test MAE:' in result.stdout:
                print("   ✅ 训练输出包含关键指标")
                return True
            else:
                print("   ⚠️ 训练完成但未找到关键指标")
                return True
        else:
            print(f"   ❌ 训练失败: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ 训练超时")
        # 删除测试配置
        test_config_path.unlink(missing_ok=True)
        return False
    
    return True

def main():
    """主函数"""
    print("时间片段消融实验框架测试")
    print("="*60)
    
    # 测试配置文件
    config_files = [
        "config_200s.yaml",
        "config_400s.yaml"
    ]
    
    project_root = Path("/root/code/ev_soc_predict")
    
    results = {}
    
    for config_file in config_files:
        config_path = project_root / "configs" / config_file
        
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            continue
        
        success = test_single_experiment(config_path)
        results[config_file] = success
        
        # 测试间隔
        time.sleep(5)
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_success = True
    for config_file, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{config_file}: {status}")
        if not success:
            all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("✅ 所有测试通过！可以运行完整消融实验。")
    else:
        print("❌ 部分测试失败，请检查问题。")
    print("="*60)

if __name__ == "__main__":
    main()
