"""
EV SoC Prediction - Training Entry Point

Usage:
  cd /root/code/ev_soc_predict && python src/main.py --window 400
  cd /root/code/ev_soc_predict && python src/main.py --window 400 --model lstm

Flow:
  1. Load configs (model_config + train_config)
  2. Set seed & device
  3. Build DataLoader
  4. Initialize model (LSTM-Transformer or LSTM baseline)
  5. Trainer training (with early stop + checkpoint)
  6. Load best model -> test set evaluation
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import model_config
from configs.train_config import train_config
from src.dataset import SocDataset, create_dataloaders
from src.models import LSTMTransformer, LSTMBaseline, MLRBaseline
from src.trainer import Trainer


MODEL_REGISTRY = {
    "lstm_transformer": LSTMTransformer,
    "lstm": LSTMBaseline,
    "mlr": MLRBaseline,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def print_config_summary(model_name: str):
    print("=" * 60)
    print("  EV SoC Prediction - %s" % model_name.upper())
    print("=" * 60)
    mc = model_config
    print(
        "  LSTM:   dim=%d, layers=%d, bi=%s"
        % (mc.lstm_hidden_dim, mc.lstm_num_layers, mc.lstm_bidirectional)
    )
    if model_name == "lstm_transformer":
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

    parser = argparse.ArgumentParser(description="EV SoC Prediction Training")
    parser.add_argument("--window", type=str, default="400",
                        help="Window size in seconds, e.g. 200/400/600/800 (default: 400)")
    parser.add_argument("--model", type=str, default="lstm_transformer",
                        choices=["lstm_transformer", "lstm", "mlr"],
                        help="Model architecture (default: lstm_transformer)")
    args = parser.parse_args()

    model_name = args.model
    model_cls = MODEL_REGISTRY[model_name]

    train_config.data_dir = "data/train/split_{}s".format(args.window)
    train_config.model_filename = "best_model_{}s_{}.pt".format(args.window, model_name)
    print("  Data dir: %s" % train_config.data_dir)

    print_config_summary(model_name)

    # 1. Seed
    set_seed(train_config.seed)

    # 2. Device
    device = resolve_device(train_config)
    print("  Device: " + device + "\n")

    # 3. Data
    y_mean, y_std = SocDataset.compute_y_stats(train_config.data_dir)
    print("  y_main stats: mean=%.4f, std=%.4f" % (y_mean, y_std))
    train_loader, val_loader, test_loader = create_dataloaders(train_config, y_mean, y_std)
    print("  Train: %d samples" % len(train_loader.dataset))
    print("  Val:   %d samples" % len(val_loader.dataset))
    print("  Test:  %d samples\n" % len(test_loader.dataset))

    # 4. Model
    model = model_cls(model_config)
    print("  Model params: %s" % f"{model.count_params():,}")
    if model_name == "lstm_transformer":
        print("  LSTM output dim: %d" % model_config.lstm_output_dim)
    print()

    # 5. Training
    trainer = Trainer(model, train_loader, val_loader, train_config, device, y_mean, y_std)
    trainer.train()

    # 6. Test
    print()
    trainer.load_best()
    test_metrics = trainer.evaluate(test_loader, phase_name="Test")

    # 7. Summary
    print()
    print("=" * 60)
    print("  Training Complete - Test Set Results [%s]" % model_name)
    print("=" * 60)
    print(
        "  DeltaSoC(MAE):  %.4f  |  RMSE: %.4f  |  R2: %.4f  |  MAPE: %.2f%%"
        % (
            test_metrics["main_mae"],
            test_metrics["main_rmse"],
            test_metrics["main_r2"],
            test_metrics["main_mape"],
        )
    )
    print(
        "  TotalEnergy(MAE): %.4f  |  RMSE: %.4f"
        % (test_metrics["aux_mae"], test_metrics["aux_rmse"])
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
