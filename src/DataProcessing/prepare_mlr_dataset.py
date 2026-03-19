# src/Modeling/train_mlr_with_refined_soc.py

import sys
import os

# ==================== 动态添加项目根目录到 sys.path ====================
# 自动推导项目根目录（无论从哪里运行该脚本）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ===================================================================

# 现在可以安全导入 configs（使用你定义的变量名）
from configs.path_config import DATA_PROCESSED_DIR, MODEL_DIR, DATA_ALIGNED_DIR

# 其他必要导入
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib


def build_mlr_dataset_from_refined_soc(df, window_size=40, step=20, min_trip_length=50):
    """
    基于 refined_soc 构建 MLR 数据集
    """
    trips = [group.reset_index(drop=True) for _, group in df.groupby('trip_id') if len(group) >= min_trip_length]
    print(f"使用 {len(trips)} 个连续行程构建数据集")

    samples = []
    for trip in trips:
        for i in range(0, len(trip) - window_size + 1, step):
            seg = trip.iloc[i:i + window_size]
            
            # 跳过含 NaN 的片段
            if seg[['refined_soc', 'speed', 'total_current']].isna().any().any():
                continue
            
            # 提取统计特征
            feat = {
                # 车辆动态
                'mean_speed': seg['speed'].mean(),
                'std_speed': seg['speed'].std(),
                'mean_current': seg['total_current'].mean(),
                'mean_volt': seg['total_volt'].mean(),
                'mean_power': (seg['total_volt'] * seg['total_current']).mean(),
                
                # 温度
                'mean_max_temp': seg['max_temp'].mean(),
                'mean_min_temp': seg['min_temp'].mean(),
                'mean_temp_diff': (seg['max_temp'] - seg['min_temp']).mean(),
                
                # 环境
                'mean_temperature_c': seg['temperature_c'].mean(),
                'mean_humidity': seg['relative_humidity'].mean(),
                'mean_wind': seg['wind_speed_ms'].mean(),
                'mean_visibility': seg['visibility_km'].mean(),
            }
            
            # 标签：使用 refined_soc 计算 ΔSOC（正数 = 消耗）
            soc_start = seg['refined_soc'].iloc[0]
            soc_end = seg['refined_soc'].iloc[-1]
            feat['delta_soc'] = soc_start - soc_end  # %
            
            samples.append(feat)
    
    return pd.DataFrame(samples)


def time_based_train_test_split(df, test_ratio=0.2):
    """
    按时间顺序划分训练/测试集（非随机！）
    """
    df_sorted = df.sort_index()
    n_test = int(len(df_sorted) * test_ratio)
    split_idx = len(df_sorted) - n_test
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    return train_df, test_df


def main():
    # === 1. 加载精细化 SOC 数据 ===
    input_file = os.path.join(DATA_PROCESSED_DIR, "aligned_data_refined_soc.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到文件: {input_file}\n请先运行 refine_soc 脚本生成数据。")
    
    df = pd.read_csv(input_file)
    df['DATE'] = pd.to_datetime(df['DATE'])
    print(f"✅ 加载 {len(df)} 行 refined_soc 数据")

    # === 2. 构建 MLR 数据集 ===
    dataset = build_mlr_dataset_from_refined_soc(
        df, 
        window_size=40,   # 400s
        step=20,          # 半重叠
        min_trip_length=60
    )
    print(f"✅ 构建 {len(dataset)} 个样本，{dataset.shape[1]-1} 个特征")

    # 保存数据集
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    dataset_path = os.path.join(DATA_PROCESSED_DIR, "mlr_dataset_refined.csv")
    dataset.to_csv(dataset_path, index=False)
    print(f"💾 数据集已保存至: {dataset_path}")

    # === 3. 划分训练/测试集（按时间顺序）===
    train_df, test_df = time_based_train_test_split(dataset, test_ratio=0.2)
    print(f"📊 训练集: {len(train_df)}，测试集: {len(test_df)}")

    # === 4. 准备 X, y ===
    feature_cols = [col for col in dataset.columns if col != 'delta_soc']
    X_train = train_df[feature_cols]
    y_train = train_df['delta_soc']
    X_test = test_df[feature_cols]
    y_test = test_df['delta_soc']

    # === 5. 训练 MLR ===
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # === 6. 评估 ===
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("📊 MLR 模型评估结果（基于 refined_soc）")
    print(f"MAE: {mae:.4f} %SOC")
    print(f"RMSE: {rmse:.4f} %SOC")
    print(f"R²: {r2:.4f}")
    print("="*50)

    # === 7. 特征重要性（系数）===
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\n🔍 MLR 特征系数（绝对值排序）:")
    print(coef_df.to_string(index=False))

    # === 8. 可视化预测 vs 真实 ===
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=10, edgecolors='none')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True ΔSOC (%)')
    plt.ylabel('Predicted ΔSOC (%)')
    plt.title('MLR: True vs Predicted ΔSOC (refined_soc)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # === 9. 保存模型 ===
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mlr_model_refined.pkl")
    joblib.dump(model, model_path)
    print(f"💾 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()