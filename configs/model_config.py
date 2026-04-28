"""
LSTM-Transformer 混合模型配置
"""


class ModelConfig:
    """模型超参数（LSTM → Transformer 串联）"""

    # LSTM
    lstm_input_dim = 17
    lstm_hidden_dim = 128
    lstm_num_layers = 2
    lstm_dropout = 0.2
    lstm_bidirectional = True

    # Transformer
    d_model = 128
    nhead = 8
    transformer_num_layers = 3
    transformer_dim_feedforward = 512
    transformer_dropout = 0.1

    # Pooling
    pooling = "mean"  # "mean" | "last"

    @property
    def lstm_output_dim(self):
        return self.lstm_hidden_dim * (2 if self.lstm_bidirectional else 1)


model_config = ModelConfig()
