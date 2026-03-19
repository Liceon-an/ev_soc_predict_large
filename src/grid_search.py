import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import pandas as pd
from pathlib import Path
import time

# =======================
# 1. 路径与环境设置
# =======================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))

import trainer
from Modeling.lstm_model import SocLSTM


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 2. 定义超参数网格
# =======================
# 在这里定义你想要搜索的参数范围
# 注意：组合总数 = 各列表长度的乘积。如果太多，训练会很慢！
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'dropout': [0.1, 0.2],
    # 'batch_size': [16, 32], # 如果显存允许，也可以加入 batch_size
}

# 训练配置 (网格搜索时通常减少 epoch 数以节省时间)
SEARCH_CONFIG = {
    'epochs': 30,          # 正式训练用 108，搜索时用 30 即可看出趋势
    'patience': 5,         # 早停耐心值调小，加快单组实验速度
    'batch_size': 32,
    'validation_split': 0.15
}

def run_single_experiment(params, train_loader, val_loader, device):
    """
    使用给定参数训练一次模型并返回验证集 Loss
    """
    h_size = params['hidden_size']
    n_layers = params['num_layers']
    lr = params['learning_rate']
    dropout = params['dropout']
    
    # 1. 初始化模型
    # 假设 input_size 已从数据中获取 (这里需要根据实际数据形状动态调整，稍后在 main 中传入)
    # 为了通用性，我们在 main 函数中实例化模型并传入 input_size
    pass 

def execute_grid_search(train_loader, val_loader, input_size, device):
    """
    执行网格搜索主逻辑
    """
    keys = param_grid.keys()
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    total_runs = len(combinations)
    logger.info(f"🔍 开始网格搜索，共 {total_runs} 种参数组合...")
    
    results = []
    best_val_loss = float('inf')
    best_params = None
    best_model_state = None
    
    start_time_total = time.time()
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        logger.info(f"\n[{i+1}/{total_runs}] 尝试参数: {params}")
        
        # 1. 构建模型
        model = SocLSTM(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1,
            dropout=params['dropout']
        ).to(device)
        
        # 2. 准备训练配置 (动态注入学习率)
        current_train_config = {
            'epochs': SEARCH_CONFIG['epochs'],
            'learning_rate': params['learning_rate'],
            'patience': SEARCH_CONFIG['patience'],
            'batch_size': SEARCH_CONFIG['batch_size'] # 实际上 batch_size 在 loader 里定了，这里主要传 lr
        }
        
        # 3. 临时保存路径 (避免覆盖主模型)
        temp_save_path = ROOT_DIR / f"temp_model_{i}.pth"
        
        # 4. 训练 (复用 trainer.train_model，但需要它支持返回最终 val_loss)
        # 注意：目前的 trainer.train_model 可能只保存模型，不返回 loss。
        # 我们需要稍微修改调用方式或读取日志。
        # 为了简单，这里我们假设 trainer 能够正常工作，我们通过重载 callback 或读取返回值来获取 loss。
        # *改进方案*：直接在这里写一个简化的训练循环，或者修改 trainer 使其返回 best_val_loss。
        
        # --- 简化版训练循环 (为了不修改 trainer.py) ---
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = torch.nn.L1Loss() # MAE
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_epoch_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(SEARCH_CONFIG['epochs']):
            # Train
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early Stopping Logic (简化版)
            if val_loss < best_epoch_loss:
                best_epoch_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy() # 保存最佳权重
            else:
                patience_counter += 1
            
            if patience_counter >= SEARCH_CONFIG['patience']:
                # logger.info(f"  Epoch [{epoch+1}] 早停触发")
                break
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch [{epoch+1}/{SEARCH_CONFIG['epochs']}] Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        # 记录结果
        result_entry = {
            'rank': i+1,
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'learning_rate': params['learning_rate'],
            'dropout': params['dropout'],
            'best_val_loss': best_epoch_loss,
            'params_str': str(params)
        }
        results.append(result_entry)
        
        logger.info(f"  ✅ 完成。最佳验证 Loss: {best_epoch_loss:.4f}")
        
        # 更新全局最佳
        if best_epoch_loss < best_val_loss:
            best_val_loss = best_epoch_loss
            best_params = params
            # best_model_state 已经在上面更新了，但那是局部变量，这里需要重新引用或保存
            # 为了安全，我们在循环外重新实例化并加载，或者直接保存文件
            torch.save(best_model_state, ROOT_DIR / "best_grid_search_model.pth")
            logger.info(f"  🏆 发现新的全局最佳! Loss: {best_val_loss:.4f}")
        
        # 清理临时状态
        del model
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    total_time = time.time() - start_time_total
    logger.info(f"\n🎉 网格搜索完成！总耗时: {total_time/60:.2f} 分钟")
    
    # 生成报告
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='best_val_loss').reset_index(drop=True)
    df_results['rank'] = range(1, len(df_results)+1)
    
    print("\n" + "="*30)
    print("📊 网格搜索结果排名 (Top 5)")
    print("="*30)
    print(df_results[['rank', 'hidden_size', 'num_layers', 'learning_rate', 'dropout', 'best_val_loss']].head(10).to_string(index=False))
    
    # 保存为 CSV
    csv_path = ROOT_DIR / "grid_search_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"详细结果已保存至: {csv_path}")
    
    return best_params, best_val_loss

# =======================
# 3. 主入口
# =======================
if __name__ == "__main__":
    # 加载数据 (复用你现有的数据加载逻辑)
    # 假设 data/processed/lstm_400_data.npz 存在
    data_path = ROOT_DIR / "data" / "processed" / "lstm_400_data.npz"
    
    if not data_path.exists():
        logger.error(f"数据文件未找到: {data_path}")
        sys.exit(1)
        
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # 转换为 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=SEARCH_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=SEARCH_CONFIG['batch_size'], shuffle=False)
    
    input_size = X_train.shape[2]
    logger.info(f"数据加载完成。Input Size: {input_size}, Train Samples: {len(X_train)}, Val Samples: {len(X_val)}")
    
    # 执行搜索
    best_params, best_loss = execute_grid_search(train_loader, val_loader, input_size, DEVICE)
    
    logger.info(f"\n💡 建议的最佳参数组合: {best_params}")
    logger.info(f"   对应的验证 Loss: {best_loss:.4f}")
    logger.info(f"请将此配置更新到 configs/config_400s.yaml 并进行完整训练 (epochs=108)。")