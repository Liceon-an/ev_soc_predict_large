import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, config, device, save_path):
    criterion = nn.L1Loss() # 使用 MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info("开始训练...")
    
    for epoch in range(config['epochs']):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{config['epochs']}] Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Early Stopping & Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  -> ✅ 模型已保存 (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"🛑 早停触发：连续 {config['patience']} 个 epoch 未提升。")
                break
                
    return history