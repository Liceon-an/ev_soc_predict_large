"""
LSTM → Transformer 串联混合模型

架构:
  Input (B, T, F)
    → LSTM (捕获时序依赖)
    → Linear Projection + LayerNorm (对齐到 d_model)
    + PositionalEncoding
    → TransformerEncoder (自注意力捕获长程依赖)
    → Global Mean Pooling
    → y_main (ΔSoC) + y_aux (总能量, 仅监控)
"""

import math

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[: x.size(1)]


class LSTMTransformer(nn.Module):
    """LSTM → Transformer 混合模型"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ---------- LSTM ----------
        self.lstm = nn.LSTM(
            input_size=cfg.lstm_input_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0,
            bidirectional=cfg.lstm_bidirectional,
        )

        # ---------- Projection: LSTM 输出 → d_model ----------
        self.proj = nn.Sequential(
            nn.Linear(cfg.lstm_output_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.transformer_dropout),
        )

        # ---------- Positional Encoding ----------
        self.pos_encoder = PositionalEncoding(cfg.d_model)

        # ---------- Transformer ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.transformer_dim_feedforward,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_num_layers,
        )

        # ---------- Prediction Heads ----------
        self.main_head = nn.Linear(cfg.d_model, 1)  # ΔSoC
        self.aux_head = nn.Linear(cfg.d_model, 1)  # 总能量 (仅监控)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
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
            y_main: (B,)  ΔSoC — 训练目标
            y_aux:  (B,)  总能量 — 仅监控
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_output_dim)

        # Projection
        proj_out = self.proj(lstm_out)  # (B, T, d_model)

        # Positional Encoding
        pos_out = self.pos_encoder(proj_out)  # (B, T, d_model)

        # Transformer
        transformer_out = self.transformer(pos_out)  # (B, T, d_model)

        # Pooling
        if self.cfg.pooling == "mean":
            pooled = transformer_out.mean(dim=1)  # (B, d_model)
        elif self.cfg.pooling == "last":
            pooled = transformer_out[:, -1, :]  # (B, d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")

        # Heads
        y_main = self.main_head(pooled).squeeze(-1)  # (B,)
        y_aux = self.aux_head(pooled).squeeze(-1)  # (B,)

        return y_main, y_aux

    def count_params(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
