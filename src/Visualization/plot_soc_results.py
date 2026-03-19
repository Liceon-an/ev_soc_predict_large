import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================
# 1. 路径与配置 (用户指定)
# =======================
# 设置中文字体 (防止乱码)
# Mac: Arial Unicode MS, Windows: SimHei, Linux: DejaVu Sans
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

# 定义固定路径
CSV_PATH = Path("/Users/liceon/Documents/ev_soc_predict/logs/SOC_LSTM_Transformer_v1/test_predictions.csv")
OUTPUT_DIR = Path(__file__).resolve().parent  # 即 src/Visualization/
OUTPUT_IMAGE = OUTPUT_DIR / "soc_analysis_report.png"

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =======================
# 2. 数据加载与后处理
# =======================
def load_and_process_data(file_path):
    """
    加载 CSV 并计算累积 SOC。
    逻辑：
    1. 读取 True_SOC_Delta 和 Pred_SOC_Delta。
    2. 累加得到绝对 SOC。
    3. 限制在 0-100 之间。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"❌ 文件未找到: {file_path}")
    
    logger.info(f"📂 正在加载文件: {file_path}")
    df = pd.read_csv(file_path)
    
    # 兼容列名检查
    col_map = {}
    if 'True_SOC_Delta' in df.columns and 'Pred_SOC_Delta' in df.columns:
        col_map = {'true': 'True_SOC_Delta', 'pred': 'Pred_SOC_Delta'}
    elif 'True' in df.columns and 'Pred' in df.columns:
        col_map = {'true': 'True', 'pred': 'Pred'}
        logger.warning("⚠️ 使用备用列名 'True' 和 'Pred'")
    else:
        raise ValueError(f"❌ 无法识别列名。当前列: {df.columns.tolist()}")

    true_delta = df[col_map['true']].values
    pred_delta = df[col_map['pred']].values

    # --- 关键步骤：从 Delta 还原为绝对 SOC ---
    # ⚠️ 重要：这里需要设置测试集的起始 SOC。
    # 如果你知道测试集第一个样本的真实绝对 SOC，请替换下面的 50.0。
    # 如果不确定，通常可以设为 50% 或从原始数据集中读取。
    INITIAL_SOC = 50.0 
    
    # 累加计算
    true_abs = INITIAL_SOC + np.cumsum(true_delta)
    pred_abs = INITIAL_SOC + np.cumsum(pred_delta)
    
    # 物理约束 (0-100%)
    true_abs = np.clip(true_abs, 0, 100)
    pred_abs = np.clip(pred_abs, 0, 100)
    
    # 构建新的 DataFrame 用于绘图
    result_df = pd.DataFrame({
        'Step': np.arange(len(df)),
        'True_SOC': true_abs,
        'Pred_SOC': pred_abs,
        'Error': pred_abs - true_abs
    })
    
    return result_df

# =======================
# 3. 绘图核心逻辑
# =======================
def plot_results(df, save_path):
    # 确保输出目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 计算指标
    mae = mean_absolute_error(df['True_SOC'], df['Pred_SOC'])
    rmse = np.sqrt(mean_squared_error(df['True_SOC'], df['Pred_SOC']))
    r2 = r2_score(df['True_SOC'], df['Pred_SOC'])
    
    logger.info(f"📊 评估结果 (绝对 SOC): MAE={mae:.4f}%, RMSE={rmse:.4f}%, R²={r2:.4f}")

    # 创建画布 (3行1列)
    fig, axs = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('LSTM-Transformer SOC 预测结果可视化分析', fontsize=16, fontweight='bold', y=0.98)

    steps = df['Step'].values
    
    # --- 图 1: SOC 跟踪曲线 ---
    ax1 = axs[0]
    ax1.plot(steps, df['True_SOC'], label='真实 SOC (Ground Truth)', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax1.plot(steps, df['Pred_SOC'], label='预测 SOC (Predicted)', color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9)
    ax1.fill_between(steps, df['True_SOC'], df['Pred_SOC'], color='gray', alpha=0.2, label='误差区域')
    
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title(f'SOC 跟踪对比 (MAE: {mae:.2f}%)', fontsize=14, pad=10)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(-5, 105)

    # --- 图 2: 误差随时间变化 ---
    ax2 = axs[1]
    ax2.plot(steps, df['Error'], color='#2ca02c', linewidth=1.5, label='预测误差 (Pred - True)')
    ax2.axhline(0, color='black', linewidth=1, linestyle='-')
    ax2.axhline(y=mae, color='red', linestyle=':', linewidth=1.5, label=f'+MAE ({mae:.2f}%)')
    ax2.axhline(y=-mae, color='red', linestyle=':', linewidth=1.5, label=f'-MAE (-{mae:.2f}%)')
    
    ax2.set_ylabel('误差 (%)', fontsize=12)
    ax2.set_xlabel('时间步 (Time Step)', fontsize=12)
    ax2.set_title('预测误差分布', fontsize=14, pad=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- 图 3: Parity Plot (预测值 vs 真实值) ---
    ax3 = axs[2]
    scatter = ax3.scatter(df['True_SOC'], df['Pred_SOC'], c=df['Error'], cmap='coolwarm', alpha=0.7, edgecolors='w', s=30, linewidth=0.5)
    
    # 绘制理想线 y=x
    min_val = min(df['True_SOC'].min(), df['Pred_SOC'].min())
    max_val = max(df['True_SOC'].max(), df['Pred_SOC'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='理想拟合 (y=x)', linewidth=2)
    
    ax3.set_xlabel('真实 SOC (%)', fontsize=12)
    ax3.set_ylabel('预测 SOC (%)', fontsize=12)
    ax3.set_title(f'Parity Plot (R² = {r2:.4f})', fontsize=14, pad=10)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('误差值 (%)', rotation=270, labelpad=15)

    # 保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"💾 报告已保存至: {save_path}")
    plt.show()

# =======================
# 4. 主执行入口
# =======================
if __name__ == "__main__":
    try:
        # 1. 加载数据
        df_processed = load_and_process_data(CSV_PATH)
        
        # 2. 绘图并保存
        plot_results(df_processed, OUTPUT_IMAGE)
        
        logger.info("✅ 绘图任务完成！")
        
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()