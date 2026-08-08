"""Foundational pipeline contracts.

The stateless primitives shared across the ``pipeline`` package:

* ``Operator`` — an Arrow -> Arrow transform.
* ``ColumnEncoder`` — the terminal Arrow -> ndarray encoder.
* ``OperatorBuilder`` — ColumnSchema -> Operator | None factory contract.
* ``ColumnPipeline`` — a frozen operator chain plus exactly one terminal
  encoder for a single column.
* ``EncodeError`` — pipeline failure carrying (table, column) context.

Concrete encoders live in ``pipeline.encoder``, null-fill operators in
``pipeline.nullfill``, and the assembly layer (``PipelineBuilder``,
``FeatureConverter``) in ``pipeline.converter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
import pyarrow as pa
from data.schema import ColumnSchema


class ColumnEncoder(Protocol):
    def encode(self, arr: pa.Array) -> np.ndarray: ...


class Operator(Protocol):
    def apply(self, arr: pa.Array) -> pa.Array: ...


class OperatorBuilder(Protocol):
    """Produce an Operator configured for one column.

    Returns ``None`` when the step does not apply to the column;
    ``PipelineBuilder`` drops those so pipeline assembly stays branch-free.
    """

    def __call__(self, col: ColumnSchema) -> Optional[Operator]: ...


class EncodeError(RuntimeError):
    """A column pipeline failed at apply time.

    Raised by ``FeatureConverter`` so the offending ``(table, column)`` is
    named in the message instead of surfacing as a bare Arrow/numpy error.
    """


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
