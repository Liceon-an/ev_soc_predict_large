import numpy as np
import torch
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def calculate_mape_safe(y_true, y_pred, threshold=0.05):
    mask = np.abs(y_true) > threshold
    if np.sum(mask) == 0:
        logger.warning(f"没有样本的真实值绝对值大于 {threshold}，无法计算有意义的 MAPE。")
        return 0.0
    
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
    return mape

def debug_mape_distribution(targets, preds):
    ranges = [
        (0, 0.5, "微小 (0-0.5%)"),
        (0.5, 1.5, "中等 (0.5-1.5%)"),
        (1.5, 3.0, "剧烈 (>1.5%)")
    ]
    logger.info("\n--- 分段 MAPE 分布 ---")
    for low, high, name in ranges:
        mask = (np.abs(targets) >= low) & (np.abs(targets) < high)
        if np.sum(mask) == 0: continue
        
        t, p = targets[mask], preds[mask]
        ape = np.abs((t - p) / t) * 100
        logger.info(f"区间 {name}: 样本数={np.sum(mask)}, 平均 APE={np.mean(ape):.2f}%, 最大 APE={np.max(ape):.2f}%")

def evaluate_and_report(model, loader, label_scaler, device, phase_name="Test"):
    model.eval()
    all_preds_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            all_preds_norm.append(outputs.cpu().numpy())
            all_targets_norm.append(y_batch.numpy())
    
    preds_norm = np.concatenate(all_preds_norm)
    targets_norm = np.concatenate(all_targets_norm)
    
    try:
        preds_real = label_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
        targets_real = label_scaler.inverse_transform(targets_norm.reshape(-1, 1)).flatten()
    except Exception as e:
        logger.error(f"反归一化失败: {e}")
        raise

    mae = mean_absolute_error(targets_real, preds_real)
    rmse = np.sqrt(mean_squared_error(targets_real, preds_real))
    r2 = r2_score(targets_real, preds_real)
    mape = calculate_mape_safe(targets_real, preds_real, threshold=0.1)
    
    # 分段分析
    mask_small = np.abs(targets_real) < 0.5
    mask_large = ~mask_small
    mae_small = mean_absolute_error(targets_real[mask_small], preds_real[mask_small]) if np.sum(mask_small) > 0 else 0
    mae_large = mean_absolute_error(targets_real[mask_large], preds_real[mask_large]) if np.sum(mask_large) > 0 else 0
    count_large = np.sum(mask_large)
    
    logger.info(f"\n=== {phase_name} 集评估报告 (真实物理量: %) ===")
    logger.info(f"样本总数: {len(targets_real)} | 显著变化样本 (>0.5%): {count_large}")
    logger.info(f"全局指标: MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}, MAPE={mape:.2f}%")
    logger.info(f"分段误差: 微小段 MAE={mae_small:.4f}%, 显著段 MAE={mae_large:.4f}%")
    
    debug_mape_distribution(targets_real, preds_real)
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'preds': preds_real, 'targets': targets_real
    }