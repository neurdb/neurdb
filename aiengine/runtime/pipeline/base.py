"""Foundational pipeline contracts.

The stateless primitives shared across the ``pipeline`` package:

* ``Operator`` — an Arrow -> Arrow transform.
* ``ColumnEncoder`` — the terminal Arrow -> ndarray encoder.
* ``ColumnPipeline`` — a frozen operator chain plus exactly one terminal
  encoder for a single column.

Concrete operators (null-fill), encoders, and the builders that assemble
pipelines from a ``ColumnSchema`` live in ``pipeline.converter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
import pyarrow as pa


class ColumnEncoder(Protocol):
    def encode(self, arr: pa.Array) -> np.ndarray: ...


class Operator(Protocol):
    def apply(self, arr: pa.Array) -> pa.Array: ...


@dataclass(frozen=True)
class Identity:
    """Pass-through operator. Safe default when a step does not apply."""

    def apply(self, arr: pa.Array) -> pa.Array:
        return arr


@dataclass(frozen=True)
class ColumnPipeline:
    operators: Tuple[Operator, ...]
    encoder: ColumnEncoder

    def apply(self, arr: pa.Array) -> np.ndarray:
        for op in self.operators:
            arr = op.apply(arr)
        return self.encoder.encode(arr)
