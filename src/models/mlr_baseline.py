"""
Multiple Linear Regression baseline for EV SoC prediction.

Architecture:
  Input (B, T, F=17)
    -> Global Mean Pooling (B, 17)
    -> Linear(17, 1) -> y_main (DeltaSoC)
    -> Linear(17, 1) -> y_aux (total energy, monitoring only)

~36 trainable parameters. No temporal modeling — pure linear baseline.
"""

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class MLRBaseline(nn.Module):
    """Multiple Linear Regression on time-averaged features."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.lstm_input_dim  # 17

        self.main_head = nn.Linear(self.input_dim, 1)  # DeltaSoC
        self.aux_head = nn.Linear(self.input_dim, 1)   # total energy

        self._init_weights()

    def _init_weights(self):
        for module in [self.main_head, self.aux_head]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, F)
        Returns:
            y_main: (B,)  DeltaSoC -- training target
            y_aux:  (B,)  total energy -- monitoring only
        """
        pooled = x.mean(dim=1)  # (B, F)

        y_main = self.main_head(pooled).squeeze(-1)  # (B,)
        y_aux = self.aux_head(pooled).squeeze(-1)    # (B,)

        return y_main, y_aux

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
