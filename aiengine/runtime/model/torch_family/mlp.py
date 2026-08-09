"""MLP over token embeddings."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..base import TaskType
from ..dispatcher import ModelDispatcher
from .base import TorchModel
from .embedding import TokenEmbedding, TorchModule

_NORMS = {
    "batch": nn.BatchNorm1d,
    "layer": nn.LayerNorm,
    "none": None,
}

_POOLINGS = {
    "mean": lambda x: x.mean(dim=1),
    "sum": lambda x: x.sum(dim=1),
}


class _MLP(nn.Module):
    """Pure architecture: X [B, N, E] -> (B, out_dim).

    Pools the field axis first (mean/sum), so the weights depend only on
    the embedding dim — never on how many fields the table has. Hidden
    blocks are standard tabular practice: Linear -> norm -> ReLU ->
    Dropout. Knows nothing about tokens, tables, or column types.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        pooling: str = "mean",
        norm: str = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(
                f"pooling must be one of {sorted(_POOLINGS)}, got {pooling!r}"
            )
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {sorted(_NORMS)}, got {norm!r}")
        self._pool = _POOLINGS[pooling]
        norm_cls = _NORMS[norm]

        in_dim = embedding_dim
        layers = []
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            if norm_cls is not None:
                layers.append(norm_cls(width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._pool(x))


@ModelDispatcher.register("mlp")
class MLPModel(TorchModel):
    """Lightweight composite box: TokenEmbedding + _MLP inside a TorchModule.

    ``params``: ``embedding_dim`` (8), ``hidden_dims`` ([64, 32]),
    ``pooling`` ("mean" | "sum"), ``norm`` ("batch" | "layer" | "none"),
    ``dropout`` (0.0), ``lr`` (1e-3).
    """

    def build_module(self) -> torch.nn.Module:
        spec = self.spec
        embedding_dim = int(self.params.get("embedding_dim", 8))
        return TorchModule(
            embedding=TokenEmbedding(self.tokenizer.vocab_size, embedding_dim),
            net=_MLP(
                embedding_dim=embedding_dim,
                out_dim=(
                    spec.n_classes
                    if spec.task_type is TaskType.MULTICLASS
                    else 1
                ),
                hidden_dims=[
                    int(h) for h in self.params.get("hidden_dims", [64, 32])
                ],
                pooling=str(self.params.get("pooling", "mean")),
                norm=str(self.params.get("norm", "batch")),
                dropout=float(self.params.get("dropout", 0.0)),
            ),
        )
