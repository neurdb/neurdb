"""FIT_ONCE family shell: boxes wrapping sklearn estimators.

The box declares only ``build_estimator(columns)``; the FIT_ONCE driver
runs ``SklearnMiddleware`` over the cached stream and calls
``estimator.fit`` exactly once — the loop lives inside sklearn.

``columns`` are the ``MatrixInput`` descriptors of the ACTUAL matrix
(post-middleware-steps), so estimator construction — e.g. which positions
to one-hot — is always consistent with the data it will fit, never a
positional convention derived from the spec. The trained estimator is
stored back on the box (``self.estimator``) for the future export round.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Tuple

from ..base import Model, ModelKind, TrainProtocol
from .middleware import MatrixColumn


class SklearnModel(Model):
    kind = ModelKind.SINGLE_TABLE
    train_protocol = TrainProtocol.FIT_ONCE

    def __init__(self, spec, params=None):
        super().__init__(spec, params)
        self.estimator: Optional[Any] = None  # set by the driver after fit

    @abstractmethod
    def build_estimator(self, columns: Tuple[MatrixColumn, ...]) -> Any:
        """Return an unfitted sklearn estimator configured from spec+params,
        adapted to the matrix described by ``columns``."""
