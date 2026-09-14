"""Learnable lookups + the general torch module wrapper.

Only LEARNABLE pieces live in the module graph — everything non-learnable
(tokenize, edge tensorization, device transfer) happened upstream in
``TorchMiddleware``. ``forward`` is pure compute.

* ``TokenEmbedding`` — one big table waiting for lookup: cell embedding =
  ``value * E[token_id]``. One gather + broadcast multiply; no column
  types, no offsets (the tokenizer froze the global vocab layout).
* ``TorchModule``   — the wrapper every torch box builds: two components,
  ``embedding`` + ``net``. ``forward(mi: ModelInput)`` selects the anchor
  table (``mi.target_table``) — works under single- or multi-table views —
  embeds, runs the pure architecture, and normalizes the head shape.
  Relational modules (future, pyg-based) compose their own wrapper over
  the same ``TokenEmbedding``, reading ``mi.edges`` as well.
"""

from __future__ import annotations

import torch
from torch import nn

from .middleware import ModelInput


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.table = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """(B, N) int64 tokens + (B, N) float32 values -> (B, N, E)."""
        return self.table(tokens) * values.unsqueeze(-1)


class TorchModule(nn.Module):
    """embedding + net; forward(ModelInput) -> output tensor.

    ``net`` is a pure architecture consuming ``X: [B, N, E]`` — what it
    does with the field axis (pool, FM interactions, attention) is its own
    business. ``y`` rides in ``mi`` for the driver's loss; by convention
    forward never reads it.
    """

    def __init__(self, embedding: TokenEmbedding, net: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.net = net

    def forward(self, mi: ModelInput) -> torch.Tensor:
        anchor = mi.target_table
        x = self.embedding(mi.tokens[anchor], mi.values[anchor])  # [B, N, E]
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
