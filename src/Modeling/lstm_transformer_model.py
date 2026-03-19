import torch
import torch.nn as nn
import math
from typing import Optional

class DynamicPositionalEncoding(nn.Module):
    """
    动态位置编码模块。
    相比固定长度，它能自动适应配置中的长序列 (如 seq_len=400)，
    并在显存允许范围内无限扩展。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(DynamicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算一个足够大的位置编码矩阵，避免每次 forward 都计算
        # 如果输入超过 max_len，会在 forward 中动态扩展
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        current_len = x.size(1)
        max_registered_len = self.pe.size(1)
        
        if current_len > max_registered_len:
            # 动态扩展位置编码 (极少发生，除非序列超长)
            device = x.device
            new_pe = torch.zeros(1, current_len, self.d_model, device=device)
            position = torch.arange(0, current_len, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
            
            new_pe[:, :, 0::2] = torch.sin(position * div_term)
            new_pe[:, :, 1::2] = torch.cos(position * div_term)
            return x + self.dropout(new_pe)
        else:
            # 使用预计算的切片
            return x + self.dropout(self.pe[:, :current_len, :])

class SocLSTMTransformer(nn.Module):
    """
    高性能混合模型：BiLSTM (局部特征) + Transformer (全局上下文)
    
    针对高算力环境优化：
    1. 支持长序列 (Seq Len 400+)
    2. 支持深层网络 (Layers 6-8) via Pre-Norm
    3. 支持 GELU 激活函数
    4. 集成 CLS Token 机制进行全局聚合
    """
    def __init__(self, input_size, hidden_size=128, num_lstm_layers=2, 
                 d_model=256, nhead=16, num_transformer_layers=6, 
                 dim_feedforward=1024, dropout=0.2, activation='gelu', 
                 output_size=1, use_cls_token=True):
        super(SocLSTMTransformer, self).__init__()
        
        self.use_cls_token = use_cls_token
        self.d_model = d_model
        
        # =======================
        # 1. LSTM 部分 (局部特征提取)
        # =======================
        # 增加 hidden_size 以匹配高算力配置
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,     # 双向捕捉前后文
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        lstm_output_dim = hidden_size * 2  # 双向
        
        # =======================
        # 2. 投影层 (维度对齐)
        # =======================
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, d_model),
            nn.LayerNorm(d_model), # 投影后加 LN 稳定训练
            nn.Dropout(dropout)
        )
        
        # =======================
        # 3. CLS Token (可选)
        # =======================
        if self.use_cls_token:
            # 初始化为较小的值，防止初期主导注意力
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # =======================
        # 4. 动态位置编码
        # =======================
        self.pos_encoder = DynamicPositionalEncoding(d_model, dropout=dropout)
        
        # =======================
        # 5. Transformer Encoder (全局上下文)
        # =======================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,      # 支持 'gelu' 或 'relu'
            batch_first=True,           # 关键：适配 LSTM 输出
            norm_first=True,            # 【核心】Pre-Norm，允许训练更深的网络
            layer_norm_eps=1e-6
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # =======================
        # 6. 回归头 (MLP Head)
        # =======================
        # 使用更深的 MLP 头以匹配大模型容量
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重，特别是 Projection 和 Head 部分"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 如果使用 CLS token，确保其已初始化
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, src, src_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: 输入张量, shape (Batch_Size, Seq_Len, Input_Features)
            src_mask: 注意力掩码 (可选). 
                      如果是 Padding mask，形状应为 (Batch, Seq_Len) 布尔值 (True 表示忽略)
                      如果是 Attention mask，形状应为 (Batch, Seq_Len, Seq_Len)
        Returns:
            output: 预测值, shape (Batch_Size, Output_Size)
        """
        B, T, F = src.shape
        
        # -------------------------------------------------
        # Step 1: LSTM 特征提取
        # -------------------------------------------------
        lstm_out, _ = self.lstm(src)  # (B, T, hidden*2)
        
        # -------------------------------------------------
        # Step 2: 线性投影 + LayerNorm + Dropout
        # -------------------------------------------------
        projected = self.projection(lstm_out)  # (B, T, d_model)
        
        # -------------------------------------------------
        # Step 3: 拼接 CLS Token
        # -------------------------------------------------
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            combined = torch.cat((cls_tokens, projected), dim=1)  # (B, T+1, d_model)
            effective_seq_len = T + 1
        else:
            combined = projected
            effective_seq_len = T
        
        # -------------------------------------------------
        # Step 4: 添加位置编码
        # -------------------------------------------------
        encoded = self.pos_encoder(combined)
        
        # -------------------------------------------------
        # Step 5: 处理 Mask (如果有)
        # -------------------------------------------------
        # 如果使用了 CLS token，需要调整 Mask 的形状
        final_mask = src_mask
        if src_mask is not None and self.use_cls_token:
            if src_mask.dim() == 2: 
                # Padding Mask: (B, T) -> (B, T+1)
                # CLS token 永远不应该被 mask 掉 (设为 False)
                cls_mask = torch.zeros(B, 1, dtype=src_mask.dtype, device=src_mask.device)
                final_mask = torch.cat([cls_mask, src_mask], dim=1)
            elif src_mask.dim() == 3:
                # Attention Mask: (B, T, T) -> (B, T+1, T+1)
                # 这里简化处理，假设用户传入的是正确的完整 mask，或者没有 mask
                # 复杂场景下需手动构建 (T+1)x(T+1) 的 mask
                pass 
        
        # -------------------------------------------------
        # Step 6: Transformer Encoder
        # -------------------------------------------------
        # is_causal=False 因为我们是回归任务，可以看到整个序列 (Bidirectional context)
        transformer_out = self.transformer_encoder(encoded, mask=final_mask, is_causal=False)
        
        # -------------------------------------------------
        # Step 7: 池化 (提取特征)
        # -------------------------------------------------
        if self.use_cls_token:
            # 提取 CLS token 的输出 (索引 0)
            pooled_features = transformer_out[:, 0, :]  # (B, d_model)
        else:
            # 全局平均池化
            pooled_features = transformer_out.mean(dim=1)  # (B, d_model)
        
        # -------------------------------------------------
        # Step 8: 回归头输出
        # -------------------------------------------------
        output = self.regression_head(pooled_features)  # (B, 1)
        
        return output.squeeze(-1)

# =======================
# 调试与验证 (模拟高算力配置)
# =======================
if __name__ == "__main__":
    print("🚀 正在初始化高算力配置模型...")
    
    # 模拟配置参数 (对应 config_lstm_transformer.yaml)
    config = {
        "input_size": 12,
        "hidden_size": 128,       # 增大
        "num_lstm_layers": 2,
        "d_model": 256,           # 增大
        "nhead": 16,              # 增多
        "num_transformer_layers": 6, # 加深
        "dim_feedforward": 1024,  # 增大
        "dropout": 0.2,
        "activation": "gelu",
        "seq_len": 400,           # 长序列测试
        "batch_size": 32
    }
    
    # 创建模拟数据
    X_dummy = torch.randn(config["batch_size"], config["seq_len"], config["input_size"])
    
    # 初始化模型
    model = SocLSTMTransformer(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_lstm_layers=config["num_lstm_layers"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_transformer_layers=config["num_transformer_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        activation=config["activation"]
    )
    
    # 移动设备 (如果有 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_dummy = X_dummy.to(device)
    
    print(f"✅ 模型结构已加载:")
    print(f"   - Transformer Layers: {config['num_transformer_layers']}")
    print(f"   - D Model: {config['d_model']}")
    print(f"   - Heads: {config['nhead']}")
    print(f"   - Seq Len Support: {config['seq_len']}+")
    
    # 前向传播测试
    model.train() # 训练模式测试 Dropout 和 BN
    
    # 模拟混合精度上下文 (可选测试)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        output = model(X_dummy)
    
    print(f"📊 输入形状: {X_dummy.shape}")
    print(f"📊 输出形状: {output.shape}")
    
    # 检查数值稳定性
    if torch.isnan(output).any():
        print("❌ 错误：输出包含 NaN！请检查初始化或学习率。")
    else:
        print("✅ 模型前向传播测试通过！数值稳定。")
        
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"💾 总参数量: {total_params / 1e6:.2f} M")