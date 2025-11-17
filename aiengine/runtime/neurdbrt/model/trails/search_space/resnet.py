from __future__ import annotations

import math
from typing import Any
from itertools import chain

from torch import Tensor
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_frame.nn.models.resnet import FCResidualBlock
from neurdbrt.model.trails.search_space.space_base import BaseSearchSpace


class TrailsResNet(Module, BaseSearchSpace):
    r"""trails:  Modified from from torch_frame.nn.models.resnet
        block_widths (list[int] | None):each residual block width，
        length must == num_layers. if it == None，then use `channels`。
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
            stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
                                | None = None,
            normalization: str | None = "layer_norm",
            dropout_prob: float = 0.2,

            block_widths: list[int] | None = None,  # ← trails added
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

        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])
        in_channels = channels * num_cols

        # ===== trails =====
        self.pre_backbone_dim = in_channels  # = channels * num_cols

        if block_widths is not None:
            if len(block_widths) != num_layers:
                raise ValueError(
                    f"`block_widths` length ({len(block_widths)}) "
                    f"must equal `num_layers` ({num_layers})."
                )
            widths = list(block_widths)
        else:
            widths = [channels] * num_layers

        # connect residual blocks：1st: in_channels -> widths[0]，following connected thereby
        blocks = []
        cur_in = in_channels
        for out_dim in widths:
            blocks.append(FCResidualBlock(
                cur_in, out_dim,
                normalization=normalization,
                dropout_prob=dropout_prob,
            ))
            cur_in = out_dim
        self.backbone = Sequential(*blocks)

        # trails: decoder fit last block width, here the channels is already the final demision.
        channels = widths[-1]
        # ==========================

        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        x = self.backbone(x)
        out = self.decoder(x)
        return out

    def forward_wo_embedding(self, x: Tensor) -> Tensor:
        """ trails
        x: [B, self.pre_backbone_dim]，this is dimension after encoder+flatten
        return: [B, out_channels]
        """
        x = self.backbone(x)
        return self.decoder(x)

    def estimate_capacity(self, include_bias: bool = True) -> int:
        """Head capacity (Linear params only; exclude encoder)."""
        n = 0
        for m in chain(self.backbone.modules(), self.decoder.modules()):
            if isinstance(m, Linear):
                n += m.in_features * m.out_features
                if include_bias and (m.bias is not None):
                    n += m.out_features
        return n

    @staticmethod
    def mutate_architecture(architecture: list[int], mutation_rate: float = 0.3) -> list[int]:
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
                mutated[i] = random.choice(TrailsResNet.channel_choices_large)
        
        return mutated

    @staticmethod
    def crossover_architectures(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
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


