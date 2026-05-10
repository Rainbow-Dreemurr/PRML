"""PyTorch positional encodings for the Transformer reproduction."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding from Vaswani et al. (2017)."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


def build_linear_absolute_table(max_len: int, d_model: int) -> torch.Tensor:
    """A deliberately simple fixed absolute code used as a baseline."""
    if max_len == 1:
        positions = torch.zeros(1, 1, dtype=torch.float32)
    else:
        positions = torch.linspace(-1.0, 1.0, max_len, dtype=torch.float32).unsqueeze(1)
    basis = torch.linspace(-1.0, 1.0, d_model, dtype=torch.float32).unsqueeze(0)
    return positions * basis


class PositionalEncoding(nn.Module):
    """Add one of several positional encodings to token embeddings."""

    def __init__(self, kind: str, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.kind = kind.lower()
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

        if self.kind == "none":
            self.learned = None
            self.register_buffer("fixed_table", torch.zeros(max_len, d_model), persistent=False)
        elif self.kind == "sinusoidal":
            self.learned = None
            self.register_buffer("fixed_table", build_sinusoidal_table(max_len, d_model), persistent=False)
        elif self.kind in {"linear", "absolute_linear"}:
            self.learned = None
            self.register_buffer("fixed_table", build_linear_absolute_table(max_len, d_model), persistent=False)
        elif self.kind in {"learned", "learned_absolute"}:
            self.learned = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.learned.weight, mean=0.0, std=0.02)
            self.register_buffer("fixed_table", torch.zeros(max_len, d_model), persistent=False)
        else:
            raise ValueError(f"Unknown position encoding kind: {kind}")

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = token_embeddings.size(1)
        x = token_embeddings * self.scale
        if self.learned is not None:
            positions = torch.arange(seq_len, device=token_embeddings.device)
            x = x + self.learned(positions).unsqueeze(0)
        else:
            x = x + self.fixed_table[:seq_len].unsqueeze(0).to(token_embeddings.device)
        return self.dropout(x)
