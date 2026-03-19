import os
import sys
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from torch.optim import AdamW  # 如果这行不存在或注释了，就会报错

# 【关键导入】混合精度训练组件
from torch.cuda.amp import autocast, GradScaler

# 【关键导入】高级学习率调度器 (需安装 transformers: pip install transformers)
try:
    from transformers import get_cosine_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ 警告: 未找到 transformers 库。高级 Warmup 调度器将不可用。请运行: pip install transformers")

# =======================
# 1. 路径与环境设置
# =======================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Modeling.lstm_transformer_model import SocLSTMTransformer

# =======================
# 2. 配置与日志初始化
# =======================
def setup_logging(log_dir, level=logging.INFO):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 高算力下可尝试开启 benchmark 加速，若结果不稳定再关闭
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False 

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# =======================
# 3. 数据加载工具 (性能优化版)
# =======================
def load_data_from_npz(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    return np.load(file_path)

def prepare_dataloaders(data, config):
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    use_amp = config['data'].get('use_amp', False) # 虽不直接影响 loader，但作为配置检查
    
    if 'X_train' in data.files:
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test = data.get('X_test', None)
        y_test = data.get('y_test', None)
    else:
        raise KeyError("数据文件中未找到 'X_train' 等键。")

    # 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    # 【关键优化】高性能 DataLoader 参数
    common_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,          # 加速 CPU->GPU 传输
        'persistent_workers': True,  # 复用 worker 进程
        'prefetch_factor': 2 if num_workers > 0 else None # 预取批次
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    
    test_loader = None
    if X_test is not None:
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    input_size = X_train.shape[2]
    logger.info(f"✅ 数据加载完成: Train={len(X_train)}, Val={len(X_val)}, Workers={num_workers}, PinMemory=True")
    
    return train_loader, val_loader, test_loader, input_size

# =======================
# 4. 模型构建
# =======================
def build_model(config, input_size, device):
    cfg = config['model']
    
    # 动态获取参数，兼容新旧配置结构
    lstm_cfg = cfg.get('lstm', {})
    trans_cfg = cfg.get('transformer', {})
    head_cfg = cfg.get('head', {})
    
    # 如果配置是扁平化的 (如新配置)，直接读取
    if 'hidden_size' in cfg:
        model = SocLSTMTransformer(
            input_size=input_size,
            hidden_size=cfg.get('hidden_size', 128),
            num_lstm_layers=cfg.get('lstm_layers', 2),
            d_model=cfg.get('d_model', 256),
            nhead=cfg.get('n_heads', 16),
            num_transformer_layers=cfg.get('transformer_layers', 6),
            dim_feedforward=cfg.get('d_ff', 1024),
            dropout=cfg.get('dropout', 0.2),
            activation=cfg.get('activation', 'gelu'),
            output_size=head_cfg.get('output_size', 1)
        )
    else:
        # 兼容旧版嵌套配置
        model = SocLSTMTransformer(
            input_size=input_size,
            hidden_size=lstm_cfg.get('hidden_size', 64),
            num_lstm_layers=lstm_cfg.get('num_layers', 2),
            d_model=trans_cfg.get('d_model', 128),
            nhead=trans_cfg.get('nhead', 4),
            num_transformer_layers=trans_cfg.get('num_layers', 2),
            dim_feedforward=trans_cfg.get('dim_feedforward', 256),
            dropout=lstm_cfg.get('dropout', 0.1),
            output_size=head_cfg.get('output_size', 1)
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🤖 模型构建完成: {total_params / 1e6:.2f} M 参数")
    
    return model

# =======================
# 5. 训练与验证核心逻辑 (AMP 增强版)
# =======================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 【关键】混合精度前向传播
        with autocast(enabled=(scaler is not None)):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        
        # 【关键】混合精度反向传播
        if scaler:
            scaler.scale(loss).backward()
            # 梯度裁剪 (需在 unscale 之后)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            # 验证时也可开启 autocast (节省显存)，但通常影响不大
            with autocast(enabled=False): 
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    return avg_loss, preds, targets

def calculate_metrics(preds, targets, epsilon=1e-6):
    """
    计算回归指标，包含 MAPE 并处理除零问题。
    
    参数:
        preds: 预测值 (numpy array 或 tensor)
        targets: 真实值 (numpy array 或 tensor)
        epsilon: 用于避免除以接近零的数的极小值阈值
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    # 确保是 numpy 数组并展平
    if hasattr(preds, 'detach'):
        preds = preds.detach().cpu().numpy()
    if hasattr(targets, 'detach'):
        targets = targets.detach().cpu().numpy()
        
    preds = preds.squeeze()
    targets = targets.squeeze()
    
    # 1. 基础指标 (MAE, RMSE, R2)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    
    # 2. 计算 MAPE (Mean Absolute Percentage Error)
    # 逻辑：
    # 由于你的数据是放电 (负值) 或 0，我们需要计算相对误差。
    # 公式：|pred - target| / |target|
    # 注意：必须排除 target 为 0 或接近 0 的情况，否则百分比会无穷大。
    
    mask = np.abs(targets) > epsilon  # 创建一个掩码，只保留绝对值大于 epsilon 的样本
    
    if np.sum(mask) == 0:
        # 如果所有样本都接近 0 (例如全是静止状态)，MAPE 无意义，设为 0 或 NaN
        mape = 0.0 
        # 或者你可以选择 mape = float('nan') 以便在日志中识别
    else:
        valid_targets = targets[mask]
        valid_preds = preds[mask]
        
        # 计算绝对百分比误差：|y_true - y_pred| / |y_true| * 100
        # 取绝对值是因为 target 是负数，我们希望得到正的百分比误差
        ape = np.abs(valid_targets - valid_preds) / np.abs(valid_targets)
        mape = np.mean(ape) * 100.0  # 转换为百分比形式 (例如 5.2 表示 5.2%)

    return {
        'MAE': mae, 
        'RMSE': rmse, 
        'R2': r2,
        'MAPE': mape,
        'Valid_MAPE_Samples': int(np.sum(mask)) # 可选：记录参与计算的样本数，方便调试
    }

# =======================
# 6. 主执行函数
# =======================
def main():
    global logger
    
    # 1. 加载配置
    config_path = CONFIGS_DIR / "config_lstm_transformer.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件未找到: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # 2. 设置日志与种子
    log_dir = ROOT_DIR / config['logging']['log_dir'] / config['experiment']['name']
    logger = setup_logging(log_dir, level=getattr(logging, config['logging'].get('level', 'INFO')))
    
    if config['experiment'].get('seed'):
        set_seed(config['experiment']['seed'])
        logger.info(f"🌱 随机种子已固定: {config['experiment']['seed']}")
    
    # 3. 设备选择
    device_str = config['experiment'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"💻 使用设备: {device}")
    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 4. 数据准备
    data_path = ROOT_DIR / config['data']['train_path']
    data = load_data_from_npz(data_path)
    train_loader, val_loader, test_loader, input_size = prepare_dataloaders(data, config)
    
    # 5. 模型构建
    model = build_model(config, input_size, device)
    
    # 6. 优化器、损失函数与调度器
    opt_cfg = config['training']
    
    # 优化器
    if opt_cfg.get('optimizer', 'adamw').lower() == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            lr=opt_cfg['base_lr'], # 适配新配置键名
            weight_decay=opt_cfg.get('weight_decay', 0.01),
            betas=opt_cfg.get('betas', (0.9, 0.999))
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg['base_lr'])
    
    # 损失函数 (支持 Huber)
    loss_type = opt_cfg.get('loss_function', 'mse').lower()
    if loss_type == 'huber':
        delta = opt_cfg.get('huber_delta', 1.0)
        criterion = nn.HuberLoss(delta=delta)
        logger.info(f"📉 使用损失函数: Huber Loss (delta={delta})")
    elif loss_type == 'mae':
        criterion = nn.L1Loss()
        logger.info("📉 使用损失函数: MAE")
    else:
        criterion = nn.MSELoss()
        logger.info("📉 使用损失函数: MSE")
        
    # 学习率调度器 (Warmup + Cosine)
    scheduler = None
    scheduler_cfg = opt_cfg.get('scheduler', {})
    num_training_steps = len(train_loader) * opt_cfg['epochs']
    
    if scheduler_cfg.get('type') == 'CosineAnnealingWarmRestarts' or scheduler_cfg.get('type') == 'cosine_warmup':
        if HAS_TRANSFORMERS:
            warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
            warmup_steps = warmup_epochs * len(train_loader)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5
            )
            logger.info(f"📅 启用调度器: Cosine Warmup (Warmup={warmup_epochs} epochs)")
        else:
            logger.warning("⚠️ 缺少 transformers 库，回退到标准 CosineAnnealingLR (无 Warmup)")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_cfg['epochs'])
    elif scheduler_cfg.get('type') == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_cfg.get('factor', 0.5), 
            patience=scheduler_cfg.get('patience', 5), min_lr=scheduler_cfg.get('min_lr', 1e-6)
        )
    
    # 混合精度 Scaler
    use_amp = config['data'].get('use_amp', False) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("⚡ 已启用自动混合精度训练 (AMP)")
    
    # 7. 训练循环
    logger.info("🚀 开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_cfg = opt_cfg.get('early_stopping', {})
    
    # TensorBoard
    writer = None
    if config['logging'].get('tensorboard', False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        
    start_time = time.time()
    clip_grad_val = opt_cfg.get('clip_grad', 1.0)
    
    for epoch in range(1, opt_cfg['epochs'] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, clip_grad=clip_grad_val)
        
        # Validate
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        val_metrics = calculate_metrics(val_preds, val_targets)
        
        # 更新调度器
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler_cfg.get('type') == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()
            
        # 日志输出
        log_msg = (f"Epoch [{epoch}/{opt_cfg['epochs']}] "
                   f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"MAE: {val_metrics['MAE']:.4f} | RMSE: {val_metrics['RMSE']:.4f} | "
                   f"R²: {val_metrics['R2']:.4f} | MAPE: {val_metrics['MAPE']:.2f}%")
        logger.info(log_msg)
        
        # TensorBoard 记录
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metrics/Val_MAE', val_metrics['MAE'], epoch)
            writer.add_scalar('Metrics/Val_RMSE', val_metrics['RMSE'], epoch) # 顺便补上 RMSE 的记录
            writer.add_scalar('Metrics/Val_R2', val_metrics['R2'], epoch)
            writer.add_scalar('Metrics/Val_MAPE', val_metrics['MAPE'], epoch) # 新增 MAPE 记录
            writer.add_scalar('LR', current_lr, epoch)
            if use_amp:
                writer.add_scalar('Misc/Grad_Scale', scaler.get_scale(), epoch)
        
        # 早停 & 保存最佳模型
        min_delta = early_stop_cfg.get('min_delta', 1e-4)
        is_improved = False
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            is_improved = True
            
            if config['logging'].get('save_checkpoint', True):
                save_path = log_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'config': config
                }, save_path)
                logger.info(f"💾 已保存最佳模型 (R²={val_metrics['R2']:.4f})")
        else:
            patience_counter += 1
            if early_stop_cfg.get('enabled', True) and patience_counter >= early_stop_cfg.get('patience', 50):
                logger.info(f"⏹️ 触发早停 (Patience={early_stop_cfg['patience']})")
                break
                
    train_time = time.time() - start_time
    logger.info(f"✅ 训练完成！总耗时: {train_time/60:.2f} 分钟")
    
    if writer:
        writer.close()
        
    # 8. 最终评估
    if test_loader is not None:
        logger.info("🧪 开始在测试集上评估...")
        ckpt_path = log_dir / "best_model.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"📥 已加载最佳模型权重 (Epoch {checkpoint['epoch']})")
        else:
            logger.warning("⚠️ 未找到最佳模型文件，使用当前权重评估。")
            
        test_loss, test_preds, test_targets = validate(model, test_loader, criterion, device)
        test_metrics = calculate_metrics(test_preds, test_targets)
        
        logger.info(f"📊 === 最终测试结果 ===")
        logger.info(f"   Loss: {test_loss:.4f}")
        logger.info(f"   MAE : {test_metrics['MAE']:.4f}")
        logger.info(f"   RMSE: {test_metrics['RMSE']:.4f}")      
        logger.info(f"   R²  : {test_metrics['R2']:.4f}")
        logger.info(f"   MAPE: {test_metrics['MAPE']:.4f}")  
        
        if config.get('evaluation', {}).get('save_predictions', False):
            pred_df = pd.DataFrame({
                'True_Value': test_targets.squeeze(),
                'Pred_Value': test_preds.squeeze(),
                'Error': np.abs(test_targets.squeeze() - test_preds.squeeze())
            })
            csv_path = log_dir / "test_predictions.csv"
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"📝 预测详情已保存至: {csv_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"❌ 训练崩溃: {e}")
        # 尝试打印显存状态以便调试 OOM
        if torch.cuda.is_available():
            logger.error(f"GPU 显存分配: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            logger.error(f"GPU 显存保留: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        sys.exit(1)