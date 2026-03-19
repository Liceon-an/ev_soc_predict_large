# src/metrics.py （新增 exclude_zeros 参数）

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred, model_name="Model", exclude_zeros=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if exclude_zeros:
        mask = np.abs(y_true) > 1e-5
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 安全计算 MAPE
    if len(y_true) > 0 and np.any(y_true != 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mape = np.nan

    return {
        "Model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "R2": float(r2),
        "Samples": len(y_true)
    }