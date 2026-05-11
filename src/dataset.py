"""
PyTorch Dataset / DataLoader 构建

从 build_dataset.py 产出的 .npz 文件加载数据。
每个样本: X=(40,14) 特征序列, y_main=总能量, y_aux=ΔSoC
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SocDataset(Dataset):
    """EV SoC 预测数据集"""

    def __init__(self, data_dir: str, split: str = "train", y_mean: float = None, y_std: float = None):
        """
        Args:
            data_dir: 数据目录 (如 data/train/split_400s)
            split: "train" | "val" | "test"
        """
        path = Path(data_dir) / f"{split}_data.npz"
        if not path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {path}")

        data = np.load(path)
        self.X = torch.from_numpy(data["X"]).float()  # (N, T=40, F=14)
        self.y_main = torch.from_numpy(data["y_main"]).float()  # (N,)
        self.y_aux = torch.from_numpy(data["y_aux"]).float()  # (N,)
        # Z-score 归一化 y_main
        if y_mean is not None and y_std is not None and y_std > 0:
            self.y_main = (self.y_main - y_mean) / y_std

    def __len__(self) -> int:
        return len(self.y_main)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_main[idx], self.y_aux[idx]

    @staticmethod
    def compute_y_stats(data_dir: str):
        """从训练集计算 y_main 的 mean/std"""
        path = Path(data_dir) / "train_data.npz"
        data = np.load(path)
        y_main = data["y_main"]
        return float(y_main.mean()), float(y_main.std())


def create_dataloaders(cfg, y_mean=None, y_std=None):
    """
    创建 train / val / test DataLoader。

    Args:
        cfg: TrainConfig 实例
    Returns:
        (train_loader, val_loader, test_loader)
    """
    common_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=0,  # Python 3.8 兼容
        pin_memory=False,
    )

    train_loader = DataLoader(
        SocDataset(cfg.data_dir, "train", y_mean=y_mean, y_std=y_std),
        shuffle=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        SocDataset(cfg.data_dir, "val", y_mean=y_mean, y_std=y_std),
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        SocDataset(cfg.data_dir, "test", y_mean=y_mean, y_std=y_std),
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader
