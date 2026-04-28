"""
EV SoC 预测 - 训练入口

用法:
  cd /root/code/ev_soc_predict && python src/main.py

流程:
  1. 加载配置 (model_config + train_config)
  2. 设置随机种子 & 设备
  3. 构建 DataLoader
  4. 初始化 LSTMTransformer 模型
  5. Trainer 训练 (含早停 + 检查点)
  6. 加载最佳模型 → 测试集评估
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import model_config
from configs.train_config import train_config
from src.dataset import SocDataset, create_dataloaders
from src.models.lstm_transformer import LSTMTransformer
from src.trainer import Trainer


def set_seed(seed: int):
    """全平台随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("  EV SoC 预测 - LSTM-Transformer")
    print("=" * 60)
    mc = model_config
    print(
        "  LSTM:   dim=%d, layers=%d, bi=%s"
        % (mc.lstm_hidden_dim, mc.lstm_num_layers, mc.lstm_bidirectional)
    )
    print(
        "  Transformer: d_model=%d, nhead=%d, nlayers=%d"
        % (mc.d_model, mc.nhead, mc.transformer_num_layers)
    )
    tc = train_config
    print(
        "  Train:  epochs=%d, batch=%d, lr=%g, wd=%g"
        % (tc.epochs, tc.batch_size, tc.lr, tc.weight_decay)
    )
    print(
        "  Early stop patience=%d, clip=%g"
        % (tc.early_stop_patience, tc.grad_clip_norm)
    )
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EV SoC 预测训练")
    parser.add_argument("--window", type=str, default="400",
                        help="窗口大小(秒)，如 200/400/600/800 (默认: 400)")
    args = parser.parse_args()

    train_config.data_dir = "data/train/split_{}s".format(args.window)
    print("  数据目录: {}".format(train_config.data_dir))

    print_config_summary()

    # 1. 随机种子
    set_seed(train_config.seed)

    # 2. 设备
    device = resolve_device(train_config)
    print("  Device: " + device + "\n")

    # 3. 数据
    y_mean, y_std = SocDataset.compute_y_stats(train_config.data_dir)
    print("  y_main stats: mean=%.4f, std=%.4f" % (y_mean, y_std))
    train_loader, val_loader, test_loader = create_dataloaders(train_config, y_mean, y_std)
    print("  Train: %d samples" % len(train_loader.dataset))
    print("  Val:   %d samples" % len(val_loader.dataset))
    print("  Test:  %d samples\n" % len(test_loader.dataset))

    # 4. 模型
    model = LSTMTransformer(model_config)
    print("  Model params: %s" % f"{model.count_params():,}")
    print("  LSTM output dim: %d" % model_config.lstm_output_dim)
    print()

    # 5. 训练
    trainer = Trainer(model, train_loader, val_loader, train_config, device, y_mean, y_std)
    trainer.train()

    # 6. 测试评估
    print()
    trainer.load_best()
    test_metrics = trainer.evaluate(test_loader, phase_name="Test")

    # 7. 最终摘要
    print()
    print("=" * 60)
    print("  训练完成 - 测试集结果")
    print("=" * 60)
    print(
        "  ΔSoC(MAE):  %.4f  |  RMSE: %.4f  |  R2: %.4f"
        % (
            test_metrics["main_mae"],
            test_metrics["main_rmse"],
            test_metrics["main_r2"],
        )
    )
    print(
        "  总能量(MAE): %.4f  |  RMSE: %.4f"
        % (test_metrics["aux_mae"], test_metrics["aux_rmse"])
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
