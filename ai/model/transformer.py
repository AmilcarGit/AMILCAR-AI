from __future__ import annotations

import torch
from torch import nn


class AMITransformer(nn.Module):
    """Transformer causal pequeño para experimentos de lenguaje desde pesos aleatorios."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible entre n_heads")
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        if length > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        return self.output(self.norm(self.blocks(x, mask=mask)))
