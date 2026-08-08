"""Terminal encoders (Arrow -> ndarray) and their schema-driven builder.

One encoder per ``ColumnStype`` — a closed 1:1 mapping:

* ``NUMERICAL``   -> :class:`NumericEncoder`
* ``CATEGORICAL`` -> :class:`CategoricalEncoder`
* ``TIMESTAMP``   -> :class:`TimestampEncoder`

All parameters (mean/std, cardinality, epoch strategy) are frozen at build
time from ``ColumnSchema.metadata`` — never inferred from incoming batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from data.base import ColumnStype, MetadataKey
from data.schema import ColumnSchema

from .base import ColumnEncoder

TimestampStrategy = Literal["epoch_s", "epoch_ms", "epoch_ns"]


@dataclass(frozen=True)
class CategoricalEncoder:
    """Map categorical values to frozen integer codes via list position.

    Values absent from ``cardinality`` — and nulls that survive null-fill —
    encode to the reserved :attr:`unknown_code` (== ``len(cardinality)``),
    so every output is a valid index. Embedding tables consuming these codes
    must therefore allocate ``len(cardinality) + 1`` rows.
    """

    cardinality: List[str]
    _value_set: pa.Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_value_set", pa.array(self.cardinality))

    @property
    def unknown_code(self) -> int:
        """Reserved out-of-vocabulary code: one past the last known value."""
        return len(self.cardinality)

    def encode(self, arr: pa.Array) -> np.ndarray:
        codes = pc.index_in(arr, value_set=self._value_set)  # type: ignore
        codes = pc.fill_null(codes, self.unknown_code)
        return codes.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


@dataclass(frozen=True)
class NumericEncoder:
    """Standardize to float32. mean=0, std=1 means pass-through cast."""

    mean: float = 0.0
    std: float = 1.0

    def encode(self, arr: pa.Array) -> np.ndarray:
        x = arr.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if self.std in (0.0, 1.0) and self.mean == 0.0:
            return x
        return (x - self.mean) / (self.std or 1.0)


@dataclass(frozen=True)
class TimestampEncoder:
    """Cast timestamp -> int64 epoch. Unit strategy pluggable."""

    strategy: Literal["epoch_s", "epoch_ms", "epoch_ns"] = "epoch_s"

    def encode(self, arr: pa.Array) -> np.ndarray:
        if not pa.types.is_timestamp(arr.type):
            raise TypeError(f"TimestampEncoder expected timestamp, got {arr.type}")
        unit = self.strategy.removeprefix("epoch_")
        return (
            pc.cast(arr, pa.timestamp(unit))
            .cast(pa.int64())
            .to_numpy(zero_copy_only=False)
        )


@dataclass
class EncoderBuilder:
    """Build a terminal encoder for a column.

    Dispatch is a closed switch over ``ColumnStype`` (adding a new stype
    requires a new enum value anyway). Subclass or pass a custom instance
    to ``FeatureConverter`` if you need different behavior.
    """

    timestamp_strategy: TimestampStrategy = "epoch_s"

    def __call__(self, col: ColumnSchema) -> Optional[ColumnEncoder]:
        stype = col.metadata.get(MetadataKey.STYPE)
        stats = col.metadata.get(MetadataKey.STATS) or {}
        if stype == ColumnStype.NUMERICAL:
            return NumericEncoder(
                mean=float(stats.get(MetadataKey.MEAN, 0.0)),
                std=float(stats.get(MetadataKey.STD, 1.0)),
            )
        if stype == ColumnStype.CATEGORICAL:
            cardinality = stats.get(MetadataKey.CARDINALITY)
            if not cardinality:
                return None
            return CategoricalEncoder(cardinality=list(cardinality))
        if stype == ColumnStype.TIMESTAMP:
            return TimestampEncoder(
                strategy=stats.get(MetadataKey.STRATEGY, self.timestamp_strategy),
            )
        return None
