import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.Visualization.plot_predictions import plot_predictions
from configs.path_config import PROCESSED_DATA_DIR, ROOT_DIR
from configs.path_config import get_path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================
# 1. 定义 LSTM 模型
# =======================
class SocLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SocLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        # 可选：增加一个中间层增强非线性
        # self.relu = nn.ReLU() 

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :] 
        # out = self.relu(out) # 如果输出有正负限制，慎用 ReLU，这里直接线性输出
        out = self.fc(out)
        return out.squeeze()

# =======================
# 2. 辅助函数：指标计算
# =======================
def calculate_mape_safe(y_true, y_pred, threshold=0.1):
    """
    计算 MAPE，但过滤掉真实值绝对值小于 threshold 的样本，防止除以零或误差爆炸。
    threshold: 最小有效变化量 (%)，默认 0.1%
    """
    mask = np.abs(y_true) > threshold
    if np.sum(mask) == 0:
        logger.warning(f"没有样本的真实值绝对值大于 {threshold}，无法计算有意义的 MAPE。")
        return 0.0
    
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
    return mape

def evaluate_and_report(model, loader, label_scaler, device, phase_name="Test"):
    """
    评估模型，执行反归一化，并计算所有指标。
    """
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
    
    # 【关键步骤】反归一化 (Inverse Transform)
    # 将标准化后的数据还原为真实的 SOC 变化量 (%)
    try:
        preds_real = label_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
        targets_real = label_scaler.inverse_transform(targets_norm.reshape(-1, 1)).flatten()
    except Exception as e:
        logger.error(f"反归一化失败: {e}")
        raise

    # 计算基础指标 (基于真实物理量)
    mae = mean_absolute_error(targets_real, preds_real)
    rmse = np.sqrt(mean_squared_error(targets_real, preds_real))
    
    # 计算 R2
    r2 = r2_score(targets_real, preds_real)
    
    # 检查方差，防止 R2 误导
    variance = np.var(targets_real)
    if variance < 1e-4:
        logger.warning(f"{phase_name} 集目标值方差过小 ({variance:.6f})，R2 指标可能不可靠 (易出现极大负值)。")

    # 计算 MAPE (带保护)
    mape = calculate_mape_safe(targets_real, preds_real, threshold=0.1)
    
    # --- 分段评估 (可选，用于深入分析) ---
    # 将数据分为 "微小变化" (|y| < 0.5) 和 "显著变化" (|y| >= 0.5)
    mask_small = np.abs(targets_real) < 0.5
    mask_large = ~mask_small
    
    mae_small = mean_absolute_error(targets_real[mask_small], preds_real[mask_small]) if np.sum(mask_small) > 0 else 0
    mae_large = mean_absolute_error(targets_real[mask_large], preds_real[mask_large]) if np.sum(mask_large) > 0 else 0
    count_large = np.sum(mask_large)
    
    # 打印报告
    logger.info(f"\n=== {phase_name} 集评估报告 (真实物理量: %) ===")
    logger.info(f"样本总数: {len(targets_real)} | 显著变化样本 (>0.5%): {count_large}")
    logger.info(f"全局指标:")
    logger.info(f"  MAE:  {mae:.4f} %")
    logger.info(f"  RMSE: {rmse:.4f} %")
    logger.info(f"  R2:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f} % (仅统计 |真实值|>0.1% 的样本)")
    logger.info(f"分段误差分析:")
    logger.info(f"  微小变化段 (|ΔSOC|<0.5%) MAE: {mae_small:.4f} %")
    logger.info(f"  显著变化段 (|ΔSOC|>=0.5%) MAE: {mae_large:.4f} %")
    
    def debug_mape_distribution(targets, preds):
        ranges = [
            (0, 0.5, "微小 (0-0.5%)"),
            (0.5, 1.5, "中等 (0.5-1.5%)"),
            (1.5, 3.0, "剧烈 (>1.5%)")
        ]
        
        for low, high, name in ranges:
            mask = (np.abs(targets) >= low) & (np.abs(targets) < high)
            if np.sum(mask) == 0: continue
            
            t, p = targets[mask], preds[mask]
            # 避免除以零
            ape = np.abs((t - p) / t) * 100
            print(f"区间 {name}: 样本数={np.sum(mask)}, 平均 APE={np.mean(ape):.2f}%, 最大 APE={np.max(ape):.2f}%")

    debug_mape_distribution(targets_real, preds_real)
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'preds': preds_real, 'targets': targets_real
    }

# =======================
# 3. 主训练流程
# =======================
def main():
    # 1. 加载配置与数据
    logger.info("正在加载数据和 Scaler...")
    
    data_path = PROCESSED_DATA_DIR / "lstm_400_data.npz"
    scaler_path = PROCESSED_DATA_DIR / "lstm_400_scalers.pkl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"未找到处理后的数据文件: {data_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"未找到 Scaler 文件: {scaler_path}")

    data = np.load(data_path)
    scalers = joblib.load(scaler_path)
    label_scaler = scalers['label']
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    logger.info(f"数据加载完成。Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. 超参数配置 (可根据 config.yaml 读取，此处硬编码示例)
    INPUT_SIZE = X_train.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    DROPOUT = 0.2
    
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 10  # 早停耐心值
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {DEVICE}")
    
    # 3. 构建 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    model = SocLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(DEVICE)
    
    # 【关键修改】使用 MAE (L1Loss) 作为损失函数
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 5. 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪 (防止异常值导致梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), ROOT_DIR / "best_lstm_model.pth")
            logger.info(f"  -> 模型已保存 (Val Loss 降低至 {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"早停触发：连续 {PATIENCE} 个 epoch 验证集未提升。")
                break
    
    # 6. 最终评估
    logger.info("\n=== 开始最终评估 ===")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(ROOT_DIR / "best_lstm_model.pth"))
    
    # 评估测试集
    results = evaluate_and_report(model, test_loader, label_scaler, DEVICE, phase_name="Test")
    
    # 评估训练集 (检查欠拟合/过拟合)
    # evaluate_and_report(model, train_loader, label_scaler, DEVICE, phase_name="Train")
    
    # 7. 可视化 (可选)
    try:
        # 1. 绘制专业预测对比图 (使用你已有的 plot_predictions 函数)
        # 假设 plot_predictions 定义在文件顶部或已导入
        # 参数顺序: (真实值, 预测值, 模型名称, 保存路径)
        plot_path = ROOT_DIR / "lstm_prediction_scatter.png"
        plot_predictions(results['targets'], results['preds'], "LSTM", str(plot_path))
        logger.info(f"✅ 散点对比图已保存至: {plot_path}")

        # 2. 保存详细预测数据到 CSV
        import pandas as pd
        
        # 构建 DataFrame
        pred_df = pd.DataFrame({
            "true_delta_soc": results['targets'],
            "pred_delta_soc": results['preds'],
            "abs_error": np.abs(results['targets'] - results['preds']),
            "rel_error_percent": np.where(
                results['targets'] != 0, 
                np.abs((results['targets'] - results['preds']) / results['targets']) * 100, 
                np.nan # 避免除以零
            ),
            "category": ["Significant (>=0.5%)" if abs(x) >= 0.5 else "Tiny (<0.5%)" for x in results['targets']]
        })

        # 保存 CSV (保留6位小数，确保精度)
        csv_path = ROOT_DIR / "lstm_predictions_detail.csv"
        pred_df.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info(f"✅ 详细预测数据已保存至: {csv_path}")
        
        # 打印前几行预览
        logger.info("\n--- 数据预览 (前5行) ---")
        logger.info("\n" + pred_df.head().to_string())
        
        # 简单统计各分类样本数
        counts = pred_df['category'].value_counts()
        logger.info(f"\n--- 样本分布统计 ---")
        for cat, count in counts.items():
            logger.info(f"  {cat}: {count} 样本")

    except Exception as e:
        logger.error(f"❌ 可视化或文件保存失败: {e}", exc_info=True)

    logger.info("训练与评估全部完成。")

if __name__ == "__main__":
    main()