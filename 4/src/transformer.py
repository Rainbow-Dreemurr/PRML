"""From-scratch PyTorch Transformer used for the homework reproduction."""

from __future__ import annotations

import math

import torch
from torch import nn

from .constants import PAD_ID
from .position_encodings import PositionalEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        self.last_attention = attention.detach()
        dropped_attention = self.dropout(attention)
        context = torch.matmul(dropped_attention, v)
        return self.w_o(self._merge_heads(context))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_residual: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.dropout(self.self_attn(x, x, x, src_mask))
        x = self.norm1(x + attn_out if self.use_residual else attn_out)
        ffn_out = self.dropout(self.ffn(x))
        x = self.norm2(x + ffn_out if self.use_residual else ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_residual: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.dropout(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x + self_attn_out if self.use_residual else self_attn_out)
        cross_attn_out = self.dropout(self.cross_attn(x, memory, memory, src_mask))
        x = self.norm2(x + cross_attn_out if self.use_residual else cross_attn_out)
        ffn_out = self.dropout(self.ffn(x))
        x = self.norm3(x + ffn_out if self.use_residual else ffn_out)
        return x


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder Transformer faithful to the paper's core modules."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pe_kind: str = "sinusoidal",
        use_residual: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_residual = use_residual
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.src_positions = PositionalEncoding(pe_kind, d_model, max_len, dropout)
        self.tgt_positions = PositionalEncoding(pe_kind, d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, use_residual) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, use_residual) for _ in range(num_layers)]
        )
        self.generator = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @staticmethod
    def make_src_mask(src: torch.Tensor) -> torch.Tensor:
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
        batch, tgt_len = tgt.shape
        pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, tgt_len, tgt_len)
        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)
        x = self.src_positions(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_positions(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src)
        decoded = self.decode(tgt_in, memory, src_mask)
        return self.generator(decoded)
