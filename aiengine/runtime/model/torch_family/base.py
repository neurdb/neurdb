"""GRADIENT family shell: declarative torch boxes.

This module imports torch at module level — the optional-dependency guard
lives in the family ``__init__``, which imports the torch-free middleware
unconditionally and the torch-dependent modules under a try, so everything
here is ordinary torch code.

The box is lightweight — two components, declared not looped:

* ``build_module()``          — the trainable ``TorchModule`` (embedding +
  architecture); all non-learnable conversion happens upstream in
  ``TorchMiddleware``
* ``loss(outputs, targets)``  — default derived from ``spec.task_type``
* ``configure_optimizer(m)``  — default Adam with ``params["lr"]``

The GRADIENT driver owns every loop: middleware conversion (cached across
epochs), device placement, forward/backward/step, metric emission, and
cancellation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import Model, ModelKind, TaskType, TrainProtocol
from ..tokenizer import FeatureTokenizer


class TorchModel(Model):
    kind = ModelKind.SINGLE_TABLE
    train_protocol = TrainProtocol.GRADIENT

    def __init__(self, spec, params=None):
        super().__init__(spec, params)
        self.tokenizer = FeatureTokenizer(spec)  # sizes for build_module
        self.module: Optional[torch.nn.Module] = None  # set after training

    @abstractmethod
    def build_module(self) -> torch.nn.Module:
        """Return the un-trained TorchModule built from spec+params."""

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.spec.task_type is TaskType.BINARY:
            return F.binary_cross_entropy_with_logits(outputs, targets.float())
        if self.spec.task_type is TaskType.MULTICLASS:
            return F.cross_entropy(outputs, targets.long())
        return F.mse_loss(outputs, targets.float())

    def configure_optimizer(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            module.parameters(), lr=float(self.params.get("lr", 1e-3))
        )
