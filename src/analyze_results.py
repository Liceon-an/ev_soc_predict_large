import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

# =======================
# 基础配置（彻底关闭字体警告，仅用英文）
# =======================
# 关闭所有Matplotlib警告
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0
})
# 设置绘图风格
sns.set_style("whitegrid")

# 核心配置：指定CSV路径
USE_SPECIFIED_CSV = True
SPECIFIED_CSV_PATH = "/root/code/ev_soc_predict/lstm_predictions_detail.csv"

# 原逻辑相关配置（保留，不影响指定CSV模式）
ROOT_DIR = Path(__file__).resolve().parent.parent if not USE_SPECIFIED_CSV else None
LOGS_DIR = ROOT_DIR / "runs" / "logs" if not USE_SPECIFIED_CSV else None

def find_latest_experiment(logs_dir):
    """查找最新的实验文件夹（原逻辑保留）"""
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    experiments = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not experiments:
        raise FileNotFoundError("No experiment folders found")
    
    latest_exp = max(experiments, key=lambda p: p.stat().st_mtime)
    return latest_exp

def load_data(exp_dir):
    """加载数据（适配你的CSV列名，修复维度问题）"""
    # 读取指定CSV
    if USE_SPECIFIED_CSV:
        specified_path = Path(SPECIFIED_CSV_PATH)
        if not specified_path.exists():
            raise FileNotFoundError(f"Specified CSV file not found: {specified_path}")
        df = pd.read_csv(specified_path)
        print(f"✅ Loaded specified prediction data: {specified_path.name} ({len(df)} samples)")
        pred_file = specified_path
    else:
        # 原逻辑（保留）
        csv_files = list(exp_dir.glob("*predictions*.csv"))
        if not csv_files:
            csv_files = list(exp_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No prediction CSV files found in {exp_dir}")
        
        pred_file = csv_files[0]
        df = pd.read_csv(pred_file)
        print(f"✅ Loaded prediction data: {pred_file.name} ({len(df)} samples)")
    
    # 标准化列名（适配你的CSV列名：true_delta_soc/pred_delta_soc）
    col_map = {
        'true_delta_soc': 'True_SOC',
        'pred_delta_soc': 'Pred_SOC',
        'abs_error': 'Error',
        'step': 'Step'
    }
    df.rename(columns=col_map, inplace=True)
    
    # 关键修复：确保核心列是一维数值
    for col in ['True_SOC', 'Pred_SOC']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
    
    # 补充Error列（如果缺失则计算）
    if 'Error' not in df.columns and 'True_SOC' in df.columns and 'Pred_SOC' in df.columns:
        df['Error'] = df['True_SOC'] - df['Pred_SOC']
    
    # 补充Step列
    if 'Step' not in df.columns:
        df['Step'] = np.arange(len(df))
    
    # 移除含NaN的行
    df = df.dropna(subset=['True_SOC', 'Pred_SOC'])
    print(f"📊 Valid samples after cleaning: {len(df)}")
    
    return df, pred_file

def plot_results(df, exp_dir, metrics):
    """生成分析图表（纯英文标题，无字体警告）"""
    # 确定图表保存路径
    if USE_SPECIFIED_CSV:
        output_dir = Path(SPECIFIED_CSV_PATH).parent / "analysis_plots"
    else:
        output_dir = exp_dir / "analysis_plots"
    output_dir.mkdir(exist_ok=True)
    
    # 提取核心数据
    true_soc = df['True_SOC'].values
    pred_soc = df['Pred_SOC'].values
    errors = df['Error'].values
    
    # --- 图 1: 真实值 vs 预测值 (全局 + 局部放大) ---
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
    # 全局图
    axs[0].plot(df['Step'], true_soc, label='True ΔSOC', color='#2E86AB', linewidth=1.5, alpha=0.8)
    axs[0].plot(df['Step'], pred_soc, label='Pred ΔSOC', color='#A23B72', linewidth=1.5, alpha=0.8, linestyle='--')
    axs[0].set_title(f'Global Comparison: True vs Predicted ΔSOC\n(R²={metrics["R2"]:.4f}, MAE={metrics["MAE"]:.4f})', 
                     fontsize=14, fontweight='bold')
    axs[0].set_ylabel('ΔSOC (%)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # 局部放大图
    window_size = min(100, len(df))
    axs[1].plot(df['Step'][:window_size], true_soc[:window_size], label='True ΔSOC', color='#2E86AB', linewidth=2)
    axs[1].plot(df['Step'][:window_size], pred_soc[:window_size], label='Pred ΔSOC', color='#A23B72', linewidth=2, linestyle='--')
    axs[1].fill_between(df['Step'][:window_size], true_soc[:window_size], pred_soc[:window_size], 
                        color='gray', alpha=0.2, label='Error Gap')
    axs[1].set_title(f'Local Zoom (First {window_size} Time Steps)', fontsize=12)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('ΔSOC (%)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 图 2: 误差分布直方图 + KDE ---
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='#F18F01', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
    plt.title(f'Error Distribution (Mean={np.mean(errors):.4f}, Std={np.std(errors):.4f})', fontsize=14)
    plt.xlabel('Error (True - Pred)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "02_error_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 图 3: 散点回归图 (Parity Plot) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(true_soc, pred_soc, s=10, alpha=0.6, color='#57C5B6', edgecolors='none')
    
    # 绘制 y=x 理想拟合线
    min_val = min(true_soc.min(), pred_soc.min())
    max_val = max(true_soc.max(), pred_soc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y=x)')
    
    # 绘制实际拟合线
    z = np.polyfit(true_soc, pred_soc, 1)
    p = np.poly1d(z)
    plt.plot(true_soc, p(true_soc), "b-", linewidth=2, label=f'Fit Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    plt.title(f'Regression Scatter Plot (R²={metrics["R2"]:.4f})', fontsize=14)
    plt.xlabel('True ΔSOC (%)')
    plt.ylabel('Predicted ΔSOC (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "03_parity_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 图 4: 不同误差类别下的箱线图 ---
    if 'category' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y='Error', data=df, hue='category', palette="coolwarm", legend=False)
        plt.title('Error Distribution by ΔSOC Category', fontsize=14)
        plt.xlabel('ΔSOC Category')
        plt.ylabel('Error (True - Pred)')
        plt.axhline(0, color='black', linewidth=1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(output_dir / "04_error_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"🎨 Plots saved to: {output_dir.absolute()}")

def plot_training_curve(exp_dir):
    """绘制训练曲线（指定CSV时跳过）"""
    if USE_SPECIFIED_CSV:
        print("⚠️ Skipping training curve plot (no log file for specified CSV)")
        return
    print("⚠️ Skipping training curve plot (no log file found)")

def main():
    print("🔍 Initializing analysis process...")
    try:
        # 初始化实验目录
        exp_dir = None if USE_SPECIFIED_CSV else find_latest_experiment(LOGS_DIR)
        
        # 1. 加载数据
        df, csv_path = load_data(exp_dir)
        
        # 2. 计算评估指标
        true_soc = df['True_SOC'].values
        pred_soc = df['Pred_SOC'].values
        
        metrics = {
            'MAE': mean_absolute_error(true_soc, pred_soc),
            'RMSE': np.sqrt(mean_squared_error(true_soc, pred_soc)),
            'R2': r2_score(true_soc, pred_soc),
            'MaxError': max_error(true_soc, pred_soc)
        }
        
        # 打印评估报告
        print("\n📊 === Detailed Evaluation Report ===")
        print(f"   Sample Count : {len(df)}")
        print(f"   MAE          : {metrics['MAE']:.6f}")
        print(f"   RMSE         : {metrics['RMSE']:.6f}")
        print(f"   R²           : {metrics['R2']:.6f}")
        print(f"   Max Error    : {metrics['MaxError']:.6f}")
        
        # 3. 生成图表
        print("\n🎨 Generating analysis plots...")
        plot_results(df, exp_dir, metrics)
        
        # 4. 绘制训练曲线
        plot_training_curve(exp_dir)
        
        print("\n✅ Analysis completed! All results saved.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()