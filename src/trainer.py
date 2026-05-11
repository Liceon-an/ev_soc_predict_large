"""
训练器 — 封装训练/验证/测试流程、早停、检查点

仅 y_main (ΔSoC) 参与 loss 计算；
y_aux (总能量) 全程仅监控，不参与梯度传播。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _calculate_mape_safe(y_true, y_pred, threshold=0.05):
    """安全计算 MAPE，过滤接近零的真实值"""
    mask = np.abs(y_true) > threshold
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _log_metrics(logger, phase, metrics):
    """统一指标日志输出"""
    parts = [f"=== {phase} ==="]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    logger.info("  |  ".join(parts))


class Trainer:
    """训练器"""

    def __init__(self, model, train_loader, val_loader, cfg, device, y_mean=None, y_std=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std

        # 优化器 & 调度器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
        )

        self.loss_fn = nn.L1Loss()  # MAE

        # 状态
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0
        self.best_epoch = 0

        # 日志
        self._setup_logging()

    def _setup_logging(self):
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"train_{timestamp}.log"

        self.logger = logging.getLogger(f"Trainer_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 文件 handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # 控制台 handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(ch)

        self.logger.info("训练日志: %s", log_path)

    def _train_epoch(self):
        self.model.train()
        losses = []
        for X, y_main, _ in self.train_loader:
            X, y_main = X.to(self.device), y_main.to(self.device)

            self.optimizer.zero_grad()
            pred_main, _ = self.model(X)
            loss = self.loss_fn(pred_main, y_main)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        preds_main, targets_main = [], []
        preds_aux, targets_aux = [], []

        for X, y_main, y_aux in self.val_loader:
            X = X.to(self.device)
            p_main, p_aux = self.model(X)
            preds_main.append(p_main.cpu())
            preds_aux.append(p_aux.cpu())
            targets_main.append(y_main)
            targets_aux.append(y_aux)

        preds_main = torch.cat(preds_main).numpy()
        targets_main = torch.cat(targets_main).numpy()
        # 反标准化到原始量纲
        if self.y_std is not None:
            preds_main = preds_main * self.y_std + self.y_mean
            targets_main = targets_main * self.y_std + self.y_mean
        preds_aux = torch.cat(preds_aux).numpy()
        targets_aux = torch.cat(targets_aux).numpy()

        metrics = {
            "main_mae": mean_absolute_error(targets_main, preds_main),
            "main_rmse": float(np.sqrt(mean_squared_error(targets_main, preds_main))),
            "main_r2": float(r2_score(targets_main, preds_main)),
            "main_mape": _calculate_mape_safe(targets_main, preds_main),
            "aux_mae": mean_absolute_error(targets_aux, preds_aux),
            "aux_rmse": float(np.sqrt(mean_squared_error(targets_aux, preds_aux))),
        }
        return metrics

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("开始训练 | 模型参数量: %d", self.model.count_params())
        self.logger.info("=" * 60)
        self.logger.info(
            "Epochs=%d | Batch=%d | LR=%g | WD=%g | Patience=%d | Clip=%.1f",
            self.cfg.epochs, self.cfg.batch_size, self.cfg.lr,
            self.cfg.weight_decay, self.cfg.early_stop_patience, self.cfg.grad_clip_norm,
        )
        self.logger.info("Device: %s", self.device)
        self.logger.info("=" * 60)

        for epoch in range(1, self.cfg.epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_loss = self._train_epoch()

            # 验证
            val_metrics = self._val_epoch()
            val_main_mae = val_metrics["main_mae"]

            # 调度器
            self.scheduler.step(val_main_mae)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 日志
            self.logger.info(
                "Epoch %3d/%d | Train Loss=%.4f | Val MAE(main)=%.4f | Val MAE(aux)=%.4f | LR=%.2e",
                epoch, self.cfg.epochs, train_loss, val_main_mae,
                val_metrics["aux_mae"], current_lr,
            )

            # 检查点
            if val_main_mae < self.best_val_mae:
                self.best_val_mae = val_main_mae
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_main_mae)
                self.logger.info("  ✓ 新最佳模型保存 (MAE=%.4f)", val_main_mae)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.early_stop_patience:
                    self.logger.info(
                        "  ⏹ 早停触发，连续 %d 轮未改善",
                        self.cfg.early_stop_patience,
                    )
                    break

        self.logger.info("=" * 60)
        self.logger.info(
            "训练结束 | 最佳轮次: %d | 最佳 Val MAE: %.4f",
            self.best_epoch, self.best_val_mae,
        )
        self.logger.info("=" * 60)

    def _save_checkpoint(self, epoch, val_mae):
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / getattr(self.cfg, "model_filename", "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_mae": val_mae,
            "cfg": self.cfg,
        }, path)

    def load_best(self):
        path = Path(self.cfg.checkpoint_dir) / getattr(self.cfg, "model_filename", "best_model.pt")
        if not path.exists():
            self.logger.warning("最佳模型文件不存在: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.logger.info("已加载最佳模型 (Epoch %d, Val MAE=%.4f)", ckpt["epoch"], ckpt["val_mae"])

    @torch.no_grad()
    def evaluate(self, loader, phase_name="Test"):
        """在给定 DataLoader 上评估模型"""
        self.model.eval()
        preds_main, targets_main = [], []
        preds_aux, targets_aux = [], []

        for X, y_main, y_aux in loader:
            X = X.to(self.device)
            p_main, p_aux = self.model(X)
            preds_main.append(p_main.cpu())
            preds_aux.append(p_aux.cpu())
            targets_main.append(y_main)
            targets_aux.append(y_aux)

        preds_main = torch.cat(preds_main).numpy()
        targets_main = torch.cat(targets_main).numpy()
        # 反标准化到原始量纲
        if self.y_std is not None:
            preds_main = preds_main * self.y_std + self.y_mean
            targets_main = targets_main * self.y_std + self.y_mean
        preds_aux = torch.cat(preds_aux).numpy()
        targets_aux = torch.cat(targets_aux).numpy()

        def _report(name, preds, targets):
            mae = mean_absolute_error(targets, preds)
            rmse = float(np.sqrt(mean_squared_error(targets, preds)))
            r2 = float(r2_score(targets, preds))
            mape = _calculate_mape_safe(targets, preds)
            self.logger.info(
                "  %s | MAE=%.4f | RMSE=%.4f | R²=%.4f | MAPE=%.2f%%",
                name, mae, rmse, r2, mape,
            )
            return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

        self.logger.info("\n========== %s 集最终评估 ==========", phase_name)
        main_metrics = _report("ΔSoC(main)", preds_main, targets_main)
        aux_metrics = _report("总能量(aux)", preds_aux, targets_aux)

        return {**{"main_" + k: v for k, v in main_metrics.items()},
                **{"aux_" + k: v for k, v in aux_metrics.items()}}
