"""Null-fill operators (Arrow -> Arrow) and their schema-driven builder.

Strategies self-register on :class:`NullFillBuilder`:

* a parameterless ``Operator`` class registers directly —
  ``@NullFillBuilder.register("forward")``;
* a parameterized strategy registers a factory ``(ColumnSchema) ->
  Operator | None`` that extracts its parameters from column metadata —
  see ``constant``, which reads ``stats.impute``.

Adding a new strategy is one file + one decorator; no builder edits.

Caveat — ``forward`` / ``backward`` are batch-local and row-order dependent:
the same row may encode differently depending on where the engine cut the
batch, and nulls at the batch edge (leading for forward, trailing for
backward) survive the fill. For categorical columns those survivors land on
``CategoricalEncoder.unknown_code`` by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pyarrow as pa
import pyarrow.compute as pc
from data.base import ColumnStype, MetadataKey
from data.schema import ColumnSchema

from .base import Identity, Operator

NullFillFactory = Callable[[ColumnSchema], Optional[Operator]]


class NullFillBuilder:
    """Build a null-fill operator from column metadata.

    Satisfies the ``OperatorBuilder`` contract: returns ``None`` when the
    step does not apply (no stype, no strategy mapped, or the strategy's
    required metadata is absent).
    """

    _registry: Dict[str, NullFillFactory] = {"default": lambda col: Identity()}

    _stype_nullfill: Dict[ColumnStype, str] = {
        ColumnStype.NUMERICAL: "constant",
        ColumnStype.CATEGORICAL: "forward",
        ColumnStype.TIMESTAMP: "default",
    }

    @classmethod
    def register(cls, name: str) -> Callable[[Any], Any]:
        """Register an operator class (parameterless) or a factory function."""

        def decorator(obj: Any) -> Any:
            if name in cls._registry:
                raise ValueError(f"null-fill strategy already registered: {name!r}")
            if isinstance(obj, type):
                cls._registry[name] = lambda col, _cls=obj: _cls()
            else:
                cls._registry[name] = obj
            return obj

        return decorator

    def __call__(self, col: ColumnSchema) -> Optional[Operator]:
        stype = col.metadata.get(MetadataKey.STYPE)
        if stype is None:
            return None
        strategy = self._stype_nullfill.get(stype)
        if strategy is None:
            return None
        return self._registry[strategy](col)


@dataclass(frozen=True)
class NullFillConstant:
    """Replace nulls with a fixed scalar (mean/median resolved offline)."""

    value: Any

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null(arr, pa.scalar(self.value, type=arr.type))


@NullFillBuilder.register("constant")
def _constant_from_column(col: ColumnSchema) -> Optional[Operator]:
    """Constant fill parameterized by ``stats.impute`` from the engine.

    No impute value in the schema means the step does not apply — the
    runtime never invents statistics, so nulls flow through to the encoder
    (NaN after the float cast).
    """
    stats = col.metadata.get(MetadataKey.STATS) or {}
    value = stats.get(MetadataKey.IMPUTE)
    if value is None:
        return None
    return NullFillConstant(value=value)


@NullFillBuilder.register("forward")
@dataclass(frozen=True)
class NullFillForward:
    """Carry the last observed value forward across nulls."""

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null_forward(arr)  # type: ignore


@NullFillBuilder.register("backward")
@dataclass(frozen=True)
class NullFillBackward:
    """Carry the next observed value backward across nulls."""

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null_backward(arr)  # type: ignore
