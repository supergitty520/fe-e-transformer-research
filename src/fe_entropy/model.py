"""A compact Transformer with higher-order-autograd-friendly attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_width)
        attention = scores.softmax(dim=-1)
        mixed = attention @ value
        mixed = mixed.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(mixed)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int, residual_scale: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Linear(expansion * width, width),
        )
        self.residual_scale = residual_scale

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.residual_scale * self.attention(
            self.attention_norm(hidden)
        )
        hidden = hidden + self.residual_scale * self.mlp(self.mlp_norm(hidden))
        return hidden


class TinyTransformer(nn.Module):
    """Encoder used by the offline reverse-sequence experiment."""

    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        layers: int,
        width: int,
        heads: int,
        expansion: int = 2,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Parameter(torch.empty(sequence_length, width))
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, expansion, residual_scale)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, vocab_size, bias=False)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = self.token_embedding(tokens) + self.position_embedding[: tokens.shape[1]]
        states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            states.append(hidden)
        logits = self.readout(self.final_norm(hidden))
        return logits, states


def reverse_sequence_loss(logits: Tensor, tokens: Tensor) -> Tensor:
    targets = torch.flip(tokens, dims=(1,))
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

