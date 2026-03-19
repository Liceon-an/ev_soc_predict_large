import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pandas as pd
import yaml
import trainer
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入自定义模块
from src.Modeling.lstm_model import SocLSTM
from src.trainer import train_model
from src.utils import evaluate_and_report
from src.Visualization.plot_predictions import plot_predictions
from configs.path_config import PROCESSED_DATA_DIR, ROOT_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = ROOT_DIR / "configs" / "config_400s.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. 加载配置
    config = load_config() # 加载的是上面新的 YAML 结构
    
    # 2. 构建路径 (使用新的 paths 键)
    data_path = PROCESSED_DATA_DIR / config['paths']['processed_data_file']
    scaler_path = PROCESSED_DATA_DIR / config['paths']['processed_scaler_file']
    model_save_path = ROOT_DIR / config['paths']['model_save_name']
    
    if not data_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"数据或 Scaler 文件缺失。检查: {data_path}, {scaler_path}")

    # 2. 加载数据
    logger.info("正在加载数据...")
    data = np.load(data_path)
    scalers = joblib.load(scaler_path)
    label_scaler = scalers['label']
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    logger.info(f"数据形状 -> Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 3. 准备 DataLoader
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {DEVICE}")
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), 
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), 
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), 
                             batch_size=batch_size, shuffle=False)
    
    # 4. 初始化模型
    input_size = X_train.shape[2]
    model_cfg = config['model']
    
    # 显式传入 input_size，忽略 yaml 中的 null
    model = SocLSTM(
        input_size=input_size,
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        output_size=model_cfg['output_size'],
        dropout=model_cfg['dropout']
    ).to(DEVICE)
    
    # 5. 执行训练
    train_config = {
        'epochs': config['training']['epochs'],
        'learning_rate': config['training']['learning_rate'],
        'patience': config['training']['patience']
    }
    
    trainer.train_model(model, train_loader, val_loader, train_config, DEVICE, model_save_path)
    
    # 6. 最终评估
    logger.info("\n=== 开始最终评估 ===")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    results = evaluate_and_report(model, test_loader, label_scaler, DEVICE, phase_name="Test")
    
    # 7. 可视化与保存结果
    try:
        # 绘图
        plot_path = ROOT_DIR / config['paths']['plot_save_name']
        plot_predictions(results['targets'], results['preds'], "LSTM", str(plot_path))
        logger.info(f"✅ 散点图已保存: {plot_path}")
        
        # 保存 CSV 详情
        pred_df = pd.DataFrame({
            "true_delta_soc": results['targets'],
            "pred_delta_soc": results['preds'],
            "abs_error": np.abs(results['targets'] - results['preds']),
            "rel_error_percent": np.where(
                results['targets'] != 0, 
                np.abs((results['targets'] - results['preds']) / results['targets']) * 100, 
                np.nan
            ),
            "category": ["Significant (>=0.5%)" if abs(x) >= 0.5 else "Tiny (<0.5%)" for x in results['targets']]
        })
        
        csv_path = ROOT_DIR / config['paths']['csv_save_name']
        pred_df.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info(f"✅ 详细数据已保存: {csv_path}")
        logger.info(f"\n数据预览:\n{pred_df.head().to_string()}")
        
    except Exception as e:
        logger.error(f"❌ 后处理失败: {e}", exc_info=True)

    logger.info("🎉 全部流程完成！")

if __name__ == "__main__":
    main()