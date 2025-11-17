from __future__ import annotations

from typing import Any

import torch
import torch_frame
from neurdbrt.model.trails.search_space.space_base import BaseSearchSpace
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder


class TrailsMLP(Module, BaseSearchSpace):
    r"""Modified From  torch_frame.nn.models.mlp
    hidden_dims (list[int] | None): trails provided:Per-layer hidden sizes for the MLP.
        If provided, it must have length == num_layers - 1.
        If None, uses uniform `channels` per hidden layer (original behavior).
    """

    blocks_choices = [2, 3]
    channel_choices = [64, 128, 256]

    blocks_choices_large = [2, 3, 4]
    channel_choices_large = [32, 64, 128, 256]

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        # ===== trails: customer hidden size =====
        if hidden_dims is not None:
            if len(hidden_dims) != max(num_layers - 1, 0):
                raise ValueError(
                    f"`hidden_dims` length ({len(hidden_dims)}) "
                    f"must equal `num_layers - 1` ({num_layers - 1})."
                )
            widths = hidden_dims
        else:
            widths = [channels] * max(num_layers - 1, 0)

        self.mlp = Sequential()
        in_dim = channels
        for out_dim in widths:
            self.mlp.append(Linear(in_dim, out_dim))
            if normalization == "layer_norm":
                self.mlp.append(LayerNorm(out_dim))
            elif normalization == "batch_norm":
                self.mlp.append(BatchNorm1d(out_dim))
            self.mlp.append(ReLU())
            self.mlp.append(Dropout(p=dropout_prob))
            in_dim = out_dim

        self.mlp.append(Linear(in_dim, out_channels))
        # ===== trails done =====

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for param in self.mlp:
            if hasattr(param, "reset_parameters"):
                param.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        x = torch.mean(x, dim=1)

        out = self.mlp(x)
        return out

    def forward_wo_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """trails
        x: [B, channels]，this is after encoder+mean-pool
        return: [B, out_channels]
        """
        return self.mlp(x)

    def estimate_capacity(self, include_bias: bool = True) -> int:
        """Head capacity (Linear params only; exclude encoder)."""
        n = 0
        for m in self.mlp.modules():
            if isinstance(m, Linear):
                n += m.in_features * m.out_features
                if include_bias and (m.bias is not None):
                    n += m.out_features
        return n

    @staticmethod
    def mutate_architecture(
        architecture: list[int], mutation_rate: float = 0.3
    ) -> list[int]:
        """
        Mutate an architecture by randomly changing some channels

        Args:
            architecture: Original architecture (list of channel sizes)
            mutation_rate: Probability of mutating each channel

        Returns:
            Mutated architecture
        """
        import random

        mutated = architecture.copy()

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate this channel
                mutated[i] = random.choice(TrailsMLP.channel_choices_large)

        return mutated

    @staticmethod
    def crossover_architectures(
        parent1: list[int], parent2: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Crossover two architectures to create two children

        Args:
            parent1: First parent architecture
            parent2: Second parent architecture

        Returns:
            Two child architectures
        """
        if len(parent1) != len(parent2):
            # If different lengths, return parents unchanged
            return parent1.copy(), parent2.copy()

        # Single-point crossover
        import random

        crossover_point = random.randint(1, len(parent1) - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2
