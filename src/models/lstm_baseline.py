"""
Pure LSTM baseline model for EV SoC prediction.

Architecture:
  Input (B, T, F)
    -> LSTM (capture temporal dependencies)
    -> Global Mean Pooling
    -> y_main (DeltaSoC) + y_aux (total energy, monitoring only)
"""

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class LSTMBaseline(nn.Module):
    """Pure LSTM model without Transformer encoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cfg.lstm_input_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0,
            bidirectional=cfg.lstm_bidirectional,
        )

        # Prediction Heads
        self.main_head = nn.Linear(cfg.lstm_output_dim, 1)  # DeltaSoC
        self.aux_head = nn.Linear(cfg.lstm_output_dim, 1)   # total energy

        self._init_weights()

    def _init_weights(self):
        """Xavier / orthogonal init for LSTM, Xavier for linear."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, F)
        Returns:
            y_main: (B,)  DeltaSoC -- training target
            y_aux:  (B,)  total energy -- monitoring only
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_output_dim)

        # Pooling
        if self.cfg.pooling == "mean":
            pooled = lstm_out.mean(dim=1)  # (B, lstm_output_dim)
        elif self.cfg.pooling == "last":
            pooled = lstm_out[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")

        # Heads
        y_main = self.main_head(pooled).squeeze(-1)  # (B,)
        y_aux = self.aux_head(pooled).squeeze(-1)    # (B,)

        return y_main, y_aux

    def count_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
