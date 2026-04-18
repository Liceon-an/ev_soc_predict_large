#!/usr/bin/env python3
"""
时间片段消融实验自动化框架
运行不同时间窗口长度的实验并收集结果
"""

import subprocess
import pandas as pd
import json
import yaml
from datetime import datetime
import os
import sys
import time
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation_study.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimeWindowAblationStudy:
    def __init__(self, config_paths, output_dir="results/ablation"):
        """
        初始化消融实验
        
        Args:
            config_paths: 配置文件路径列表
            output_dir: 结果输出目录
        """
        self.config_paths = config_paths
        self.output_dir = Path(output_dir)
        self.results = []
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化消融实验")
        logger.info(f"配置文件数量: {len(config_paths)}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def run_experiment(self, config_path):
        """
        运行单个实验
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 实验指标
        """
        config_path = Path(config_path)
        config_name = config_path.stem.replace('config_', '')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始实验: {config_name}")
        logger.info(f"{'='*60}")
        
        # 读取配置信息
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        experiment_name = config.get('experiment_name', config_name)
        seq_len = config['data_processing']['sequence_length']
        stride = config['data_processing']['stride']
        
        logger.info(f"实验名称: {experiment_name}")
        logger.info(f"序列长度: {seq_len} 步")
        logger.info(f"步长: {stride} 步")
        logger.info(f"时间窗口: {seq_len * 10} 秒")  # 假设10秒/步
        
        experiment_metrics = {
            'config_name': config_name,
            'experiment_name': experiment_name,
            'sequence_length': seq_len,
            'stride': stride,
            'time_window_seconds': seq_len * 10,
            'start_time': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # 1. 数据预处理
            logger.info("步骤 1/3: 数据预处理...")
            preprocess_start = time.time()
            
            preprocess_cmd = [
                "/root/miniconda3/bin/python",
                "src/DataProcessing/preprocess.py",
                "--config", str(config_path)
            ]
            
            logger.debug(f"执行命令: {' '.join(preprocess_cmd)}")
            preprocess_result = subprocess.run(
                preprocess_cmd,
                cwd="/root/code/ev_soc_predict",
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            preprocess_time = time.time() - preprocess_start
            
            if preprocess_result.returncode != 0:
                logger.error(f"数据预处理失败: {preprocess_result.stderr}")
                experiment_metrics['preprocess_error'] = preprocess_result.stderr
                return experiment_metrics
            
            logger.info(f"数据预处理完成，耗时: {preprocess_time:.1f}秒")
            
            # 解析预处理输出，获取样本数量
            sample_count = self._parse_sample_count(preprocess_result.stdout)
            experiment_metrics['sample_count'] = sample_count
            experiment_metrics['preprocess_time'] = preprocess_time
            
            # 2. 模型训练
            logger.info("步骤 2/3: 模型训练...")
            train_start = time.time()
            
            train_cmd = [
                "/root/miniconda3/bin/python",
                "src/lstm.py",
                "--config", str(config_path)
            ]
            
            logger.debug(f"执行命令: {' '.join(train_cmd)}")
            train_result = subprocess.run(
                train_cmd,
                cwd="/root/code/ev_soc_predict",
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            train_time = time.time() - train_start
            
            if train_result.returncode != 0:
                logger.error(f"模型训练失败: {train_result.stderr}")
                experiment_metrics['train_error'] = train_result.stderr
                return experiment_metrics
            
            logger.info(f"模型训练完成，耗时: {train_time:.1f}秒")
            
            # 3. 解析训练结果
            logger.info("步骤 3/3: 解析结果...")
            metrics = self._parse_training_output(train_result.stdout)
            
            experiment_metrics.update(metrics)
            experiment_metrics['train_time'] = train_time
            experiment_metrics['total_time'] = preprocess_time + train_time
            experiment_metrics['success'] = True
            experiment_metrics['end_time'] = datetime.now().isoformat()
            
            logger.info(f"实验完成: {config_name}")
            logger.info(f"测试 MAE: {metrics.get('test_mae', 'N/A'):.4f}")
            logger.info(f"测试 R²: {metrics.get('test_r2', 'N/A'):.4f}")
            logger.info(f"总耗时: {experiment_metrics['total_time']:.1f}秒")
            
        except subprocess.TimeoutExpired:
            logger.error(f"实验超时: {config_name}")
            experiment_metrics['timeout'] = True
        except Exception as e:
            logger.error(f"实验异常: {config_name}, 错误: {e}")
            experiment_metrics['exception'] = str(e)
        
        return experiment_metrics
    
    def _parse_sample_count(self, output):
        """从预处理输出中解析样本数量"""
        lines = output.split('\n')
        for line in lines:
            if '有效样本' in line or '生成有效样本数' in line:
                # 查找数字
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return int(numbers[0])
        return None
    
    def _parse_training_output(self, output):
        """从训练输出中解析指标"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 解析测试指标
            if line.startswith('Test MAE:'):
                try:
                    metrics['test_mae'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Test RMSE:'):
                try:
                    metrics['test_rmse'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Test R²:'):
                try:
                    metrics['test_r2'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Test MAPE:'):
                try:
                    metrics['test_mape'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            
            # 解析训练指标
            elif 'Best validation MAE:' in line:
                try:
                    metrics['best_val_mae'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Epoch' in line and 'val_mae' in line:
                # 尝试解析epoch信息
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'val_mae:':
                            metrics['final_val_mae'] = float(parts[i+1])
                            break
                except:
                    pass
        
        return metrics
    
    def run_all(self):
        """运行所有实验"""
        logger.info(f"\n{'#'*60}")
        logger.info(f"开始运行消融实验")
        logger.info(f"实验数量: {len(self.config_paths)}")
        logger.info(f"{'#'*60}")
        
        for i, config_path in enumerate(self.config_paths, 1):
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"配置文件不存在: {config_path}, 跳过")
                continue
            
            logger.info(f"\n实验 {i}/{len(self.config_paths)}: {config_path.name}")
            
            # 运行实验
            metrics = self.run_experiment(config_path)
            self.results.append(metrics)
            
            # 实时保存结果
            self._save_intermediate_results()
            
            # 实验间隔
            if i < len(self.config_paths):
                logger.info(f"等待10秒后开始下一个实验...")
                time.sleep(10)
        
        # 最终保存结果
        self.save_results()
        self.generate_report()
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"所有实验完成!")
        logger.info(f"{'#'*60}")
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        if not self.results:
            return
        
        intermediate_file = self.output_dir / "intermediate_results.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def save_results(self):
        """保存实验结果"""
        if not self.results:
            logger.warning("没有实验结果可保存")
            return
        
        # 保存为JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"结果已保存:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV: {csv_path}")
    
    def generate_report(self):
        """生成分析报告"""
        if not self.results:
            logger.warning("没有实验结果可生成报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 只保留成功的实验
        successful_df = df[df['success'] == True].copy()
        
        if successful_df.empty:
            logger.warning("没有成功的实验")
            return
        
        # 计算统计信息
        report = f"""
# 时间片段消融实验报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**实验数量**: {len(self.results)} (成功: {len(successful_df)}, 失败: {len(df) - len(successful_df)})

## 实验配置
| 配置名称 | 时间窗口(秒) | 序列长度 | 步长 | 样本数量 |
|----------|--------------|----------|------|----------|
"""
        
        for _, row in successful_df.iterrows():
            report += f"| {row['config_name']} | {row['time_window_seconds']} | {row['sequence_length']} | {row['stride']} | {row.get('sample_count', 'N/A')} |\n"
        
        report += f"""
## 性能结果
| 配置名称 | 测试 MAE | 测试 RMSE | 测试 R² | 训练时间(秒) | 总时间(秒) |
|----------|----------|-----------|---------|--------------|------------|
"""
        
        for _, row in successful_df.iterrows():
            report += f"| {row['config_name']} | {row.get('test_mae', 'N/A'):.4f} | {row.get('test_rmse', 'N/A'):.4f} | {row.get('test_r2', 'N/A'):.4f} | {row.get('train_time', 'N/A'):.1f} | {row.get('total_time', 'N/A'):.1f} |\n"
        
        # 找到最佳配置
        if 'test_mae' in successful_df.columns:
            best_by_mae = successful_df.loc[successful_df['test_mae'].idxmin()]
            worst_by_mae = successful_df.loc[successful_df['test_mae'].idxmax()]
            
            report += f"""
## 关键发现

### 最佳性能配置 (按测试 MAE)
- **配置**: {best_by_mae['config_name']}
- **时间窗口**: {best_by_mae['time_window_seconds']} 秒
- **测试 MAE**: {best_by_mae['test_mae']:.4f}
- **测试 R²**: {best_by_mae.get('test_r2', 'N/A'):.4f}

### 最差性能配置 (按测试 MAE)
- **配置**: {worst_by_mae['config_name']}
- **时间窗口**: {worst_by_mae['time_window_seconds']} 秒
- **测试 MAE**: {worst_by_mae['test_mae']:.4f}
- **测试 R²**: {worst_by_mae.get('test_r2', 'N/A'):.4f}

### 最快训练配置
"""
        
        if 'train_time' in successful_df.columns:
            fastest = successful_df.loc[successful_df['train_time'].idxmin()]
            report += f"- **配置**: {fastest['config_name']}\n"
            report += f"- **训练时间**: {fastest['train_time']:.1f} 秒\n"
            report += f"- **测试 MAE**: {fastest.get('test_mae', 'N/A'):.4f}\n"
        
        report += f"""
## 分析建议

1. **时间窗口选择**: 基于性能和计算成本的权衡，建议选择 {best_by_mae['config_name']} 配置
2. **样本效率**: 观察样本数量与时间窗口的关系，长窗口可能减少样本数量
3. **计算成本**: 训练时间随序列长度增加而增加，需考虑实际部署需求

## 失败实验分析
"""
        
        failed_experiments = df[df['success'] == False]
        if not failed_experiments.empty:
            for _, row in failed_experiments.iterrows():
                report += f"- **{row['config_name']}**: "
                if 'preprocess_error' in row and pd.notna(row['preprocess_error']):
                    report += "数据预处理失败\n"
                elif 'train_error' in row and pd.notna(row['train_error']):
                    report += "模型训练失败\n"
                elif 'timeout' in row and row['timeout']:
                    report += "实验超时\n"
                elif 'exception' in row and pd.notna(row['exception']):
                    report += f"异常: {row['exception'][:100]}...\n"
                else:
                    report += "未知原因\n"
        else:
            report += "所有实验均成功完成！\n"
        
        # 保存报告
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析报告已保存: {report_path}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("实验摘要")
        print("="*60)
        print(f"总实验数: {len(self.results)}")
        print(f"成功: {len(successful_df)}")
        print(f"失败: {len(df) - len(successful_df)}")
        
        if not successful_df.empty:
            print(f"\n最佳配置: {best_by_mae['config_name']} (MAE: {best_by_mae['test_mae']:.4f})")
            print(f"最差配置: {worst_by_mae['config_name']} (MAE: {worst_by_mae['test_mae']:.4f})")
        
        print(f"\n详细报告: {report_path}")

def main():
    """主函数"""
    # 项目根目录
    project_root = Path("/root/code/ev_soc_predict")
    
    # 配置文件列表
    config_files = [
        "config_200s.yaml",
        "config_400s.yaml", 
        "config_600s.yaml",
        "config_800s.yaml",
        "config_1000s.yaml"
    ]
    
    config_paths = [project_root / "configs" / f for f in config_files]
    
    # 检查配置文件是否存在
    valid_configs = []
    for config_path in config_paths:
        if config_path.exists():
            valid_configs.append(config_path)
        else:
            logger.warning(f"配置文件不存在: {config_path}")
    
    if not valid_configs:
        logger.error("没有有效的配置文件，退出")
        return
    
    # 创建消融实验实例
    study = TimeWindowAblationStudy(
        config_paths=valid_configs,
        output_dir=project_root / "results" / "time_window_ablation"
    )
    
    # 运行实验
    study.run_all()

if __name__ == "__main__":
    main()
