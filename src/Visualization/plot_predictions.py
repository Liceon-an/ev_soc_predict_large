import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 核心修改：固定指定保存到 'plot' 文件夹（优先级最高）
# 无论是否导入配置文件，最终都使用当前脚本目录下的 'plot' 文件夹
# 你也可以修改为绝对路径，比如 
PLOT_DIR = Path("/root/code/ev_soc_predict/plot")
# 强制创建 plot 文件夹（无则新建，有则不影响）
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 绘图保存目录: {PLOT_DIR} (无则自动创建)")

# 定义固定的输入文件路径 (根据你的实际路径修改！)
# 注意：Mac 路径和 Linux/Windows 路径区别，这里可根据环境调整
#DEFAULT_CSV_PATH = Path("/Users/liceon/Documents/ev_soc_predict/logs/SOC_LSTM_Transformer_v1/test_predictions.csv")
# 如果是 AutoDL 等 Linux 环境，替换为：
DEFAULT_CSV_PATH = Path("/root/code/ev_soc_predict/logs/SOC_LSTM_Transformer_v1/test_predictions.csv")

def plot_predictions(y_true, y_pred, model_name="Model", save_name=None, output_dir=None):
    """
    绘制真实值 vs 预测值散点图 + 对角线
    :param y_true: 真实 ΔSOC (array-like)
    :param y_pred: 预测 ΔSOC (array-like)
    :param model_name: 模型名称（标题用）
    :param save_name: 保存文件名（如 'mlr_pred.png'），若 None 则不保存
    :param output_dir: 输出目录路径，默认为 PLOT_DIR（固定为 plot 文件夹）
    """
    # 强制使用 plot 文件夹，覆盖传入的其他目录
    output_dir = PLOT_DIR
    # 再次确认目录存在（双重保障）
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8)) # 稍微调大图片尺寸
    plt.scatter(y_true, y_pred, alpha=0.6, s=15, c='blue', edgecolors='w', linewidth=0.5)
    
    # 绘制对角线 y=x
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
    
    plt.xlabel("True ΔSOC (%)", fontsize=12)
    plt.ylabel("Predicted ΔSOC (%)", fontsize=12)
    plt.title(f"{model_name}: Prediction vs True", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 添加 R² 和 MAE 标注 (可选，增加信息量)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    text_str = f'R² = {r2:.4f}\nMAE = {mae:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_name:
        save_path = output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图片已保存到: {save_path}")
    
    plt.show()
    plt.close()  # 避免内存累积

if __name__ == "__main__":
    # =======================
    # 主程序入口：直接读取指定 CSV 并绘图
    # =======================
    
    print(f"🔍 正在读取文件: {DEFAULT_CSV_PATH}")
    
    if not DEFAULT_CSV_PATH.exists():
        print(f"❌ 错误：文件未找到！请检查路径:\n{DEFAULT_CSV_PATH}")
        print("💡 提示：请修改 DEFAULT_CSV_PATH 为你的 CSV 文件实际路径")
        sys.exit(1)

    try:
        # 1. 读取数据
        df = pd.read_csv(DEFAULT_CSV_PATH)
        
        # 2. 自动识别列名
        col_true = None
        col_pred = None
        
        # 可能的列名组合（扩展识别范围）
        possible_pairs = [
            ('True', 'Pred'),
            ('True_SOC_Delta', 'Pred_SOC_Delta'),
            ('y_true', 'y_pred'),
            ('true_soc', 'pred_soc'),
            ('actual', 'predicted')
        ]
        
        for true_col, pred_col in possible_pairs:
            if true_col in df.columns and pred_col in df.columns:
                col_true, col_pred = true_col, pred_col
                break
        
        if not col_true or not col_pred:
            print(f"❌ 无法识别列名。可用列: {list(df.columns)}")
            print("💡 请确保 CSV 中包含以下任意列对：")
            for pair in possible_pairs:
                print(f"   - {pair[0]} / {pair[1]}")
            sys.exit(1)

        print(f"📊 检测到列: '{col_true}' 和 '{col_pred}'")

        # 提取数据 (处理可能的 NaN 和无穷值)
        df_clean = df[[col_true, col_pred]].dropna()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        y_true = df_clean[col_true].values
        y_pred = df_clean[col_pred].values
        
        if len(y_true) == 0:
            print("❌ 错误：有效数据为空（可能全是 NaN/无穷值）。")
            sys.exit(1)

        # 3. 调用绘图函数
        model_name = "LSTM-Transformer" 
        save_filename = "soc_delta_parity_plot.png"
        
        print(f"🚀 开始绘图...")
        plot_predictions(
            y_true=y_true, 
            y_pred=y_pred, 
            model_name=model_name, 
            save_name=save_filename,
            # 即使传入其他目录，也会被函数内的 PLOT_DIR 覆盖
            output_dir=None
        )
        
        print("✅ 任务完成！")
        print(f"🔍 可前往以下路径查看图片：{PLOT_DIR / save_filename}")

    except ImportError as e:
        if "sklearn" in str(e):
            print("❌ 缺少 sklearn 库，请执行：pip install scikit-learn")
        else:
            print(f"❌ 导入模块出错: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)