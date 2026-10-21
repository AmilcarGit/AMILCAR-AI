"""Transformer causal pequeño, escrito a mano (sin nn.TransformerEncoder).

Componentes: RMSNorm, posiciones rotatorias (RoPE), atención causal con caché de
claves/valores (KV cache) y una capa de alimentación SwiGLU. Pensado para entrenarse
desde pesos aleatorios.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).type_as(x) * self.weight


def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Tablas de coseno y seno con forma (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rota pares de dimensiones. x: (B, H, T, head_dim); cos y sin: (T, head_dim // 2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present: KVCache = (k, v)
        dropout_p = self.dropout if self.training else 0.0
        if past is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # Con caché hay `length` consultas nuevas y `total` claves; cada consulta ve
            # todo el pasado y las nuevas posiciones hasta la suya.
            total = k.shape[2]
            mask = torch.ones(length, total, dtype=torch.bool, device=x.device).tril(diagonal=total - length)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.proj(out), present


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        hidden = ((int(d_model * 8 / 3) + 7) // 8) * 8
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp_norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attended, present = self.attn(self.attn_norm(x), cos, sin, past)
        x = x + self.drop(attended)
        normed = self.mlp_norm(x)
        x = x + self.drop(self.down(F.silu(self.gate(normed)) * self.up(normed)))
        return x, present


class AMITransformer(nn.Module):
    """Modelo de lenguaje causal. forward devuelve logits (B, T, vocab)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible entre n_heads")
        if (d_model // n_heads) % 2:
            raise ValueError("d_model / n_heads debe ser par (lo exige RoPE)")
        if n_layers < 1 or max_seq_len < 2:
            raise ValueError("n_layers debe ser >= 1 y max_seq_len >= 2")
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        cos, sin = build_rope_cache(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        if past_key_values is None and input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
        start = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        length = input_ids.shape[1]
        if start + length > self.max_seq_len:
            raise ValueError("La secuencia supera max_seq_len; reconstruye la caché con una ventana menor.")
        x = self.token_embedding(input_ids)
        cos = self.rope_cos[start:start + length]
        sin = self.rope_sin[start:start + length]
        presents: list[KVCache] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, cos, sin, past)
            presents.append(present)
        logits = self.output(self.norm(x))
        if use_cache:
            return logits, presents
        return logits
