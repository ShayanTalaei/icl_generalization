import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import SeqModel


# ---------------------------------------------------------------------------
# Positional encoding helpers
# ---------------------------------------------------------------------------

def _sinusoidal_encoding(max_len: int, d_model: int) -> Tensor:
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _precompute_rope_freqs(d_head: int, max_len: int, base: float = 10000.0):
    """Precompute cos/sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)          # (max_len, d_head/2)
    return freqs.cos(), freqs.sin()            # each (max_len, d_head/2)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to Q or K.

    Args:
        x:   (B, n_heads, S, d_head)
        cos: (S, d_head/2)  (or broadcastable)
        sin: (S, d_head/2)
    """
    x1 = x[..., 0::2]   # (B, n_heads, S, d_head/2)
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)  # (B, n_heads, S, d_head)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor,
                rope_cos: Tensor | None = None,
                rope_sin: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, S, d_head)

        if rope_cos is not None:
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> Attn -> Add -> LN -> FFN -> Add."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor,
                rope_cos: Tensor | None = None,
                rope_sin: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CausalTransformer(SeqModel):
    """GPT-style causal Transformer for in-context learning.

    Internally builds an interleaved sequence [x1, y1, x2, y2, ..., x_query]
    with causal masking.  Predictions are read out at the x-positions.

    Supports four positional encoding modes:
        learned     -- nn.Embedding, does NOT generalise beyond training length
        sinusoidal  -- fixed Vaswani-style, generalises to any length
        rope        -- Rotary Position Embeddings applied to Q/K in attention
        none        -- no positional information at all
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        dropout: float = 0.0,
        pos_encoding: str = "sinusoidal",
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding

        self.proj_x = nn.Linear(d_in, d_model)
        self.proj_y = nn.Linear(d_out, d_model)
        self.proj_out = nn.Linear(d_model, d_out)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- positional encoding ---
        if pos_encoding == "learned":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        elif pos_encoding == "sinusoidal":
            self.register_buffer("pos_emb", _sinusoidal_encoding(max_seq_len, d_model))
        elif pos_encoding == "rope":
            d_head = d_model // n_heads
            cos, sin = _precompute_rope_freqs(d_head, max_seq_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)
        elif pos_encoding == "none":
            pass
        else:
            raise ValueError(
                f"Unknown pos_encoding: {pos_encoding!r}. "
                f"Choose from: learned, sinusoidal, rope, none"
            )

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        B, n, _ = xs.shape
        seq_len = 2 * n - 1

        x_emb = self.proj_x(xs)             # (B, n, d_model)
        y_emb = self.proj_y(ys[:, :-1, :])  # (B, n-1, d_model)

        seq = xs.new_zeros(B, seq_len, self.d_model)
        seq[:, 0::2, :] = x_emb
        seq[:, 1::2, :] = y_emb

        # Additive positional encoding (learned / sinusoidal)
        pe = self.pos_encoding_type
        if pe == "learned":
            seq = seq + self.pos_emb(torch.arange(seq_len, device=xs.device))
        elif pe == "sinusoidal":
            seq = seq + self.pos_emb[:seq_len]

        # RoPE cos/sin for this sequence length (passed into each block)
        rope_cos = rope_sin = None
        if pe == "rope":
            rope_cos = self.rope_cos[:seq_len]  # (S, d_head/2)
            rope_sin = self.rope_sin[:seq_len]

        for block in self.blocks:
            seq = block(seq, rope_cos, rope_sin)
        seq = self.final_norm(seq)

        y_preds = self.proj_out(seq[:, 0::2, :])  # (B, n, d_out)
        return y_preds
