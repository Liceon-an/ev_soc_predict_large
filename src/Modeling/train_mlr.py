# src/Modeling/train_mlr.py

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.path_config import DATA_PROCESSED_DIR, MODEL_DIR, RESULT_DIR, PLOT_DIR
from src.metrics import compute_regression_metrics
from src.Visualization.plot_predictions import plot_predictions
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    # 加载数据
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "mlr_dataset_refined.csv"))
    df = df.dropna()
    
    X = df.drop(columns=['delta_soc'])
    y = df['delta_soc']
    
    # 划分（保持与之前一致）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 训练
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 保存模型
    joblib.dump(model, os.path.join(MODEL_DIR, "mlr_model_refined.pkl"))

    # 计算指标
    metrics = compute_regression_metrics(y_test, y_pred, model_name="MLR")
    print("\n📊 MLR Evaluation:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 保存指标
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULT_DIR, "mlr_metrics.csv"), index=False
    )

    # 绘图
    plot_predictions(y_test, y_pred, "MLR", "mlr_prediction.png")

    # 保存预测结果（供后续对比）
    pred_df = pd.DataFrame({"true": y_test, "pred": y_pred})
    pred_df.to_csv(os.path.join(RESULT_DIR, "mlr_predictions.csv"), index=False)

if __name__ == "__main__":
    main()