"""DataBatch -> EncodedFeatures via per-column transform pipelines.

``FeatureConverter`` runs per-column pipelines for categorical / numeric /
timestamp encoding and returns ``EncodedFeatures`` — framework-agnostic numpy
the trainer wraps in torch/tf/jax.

Per-column work is a two-stage pipeline:

    Arrow -> (Operator chain, Arrow -> Arrow) -> Encoder (Arrow -> ndarray)

Operators are stateless: their parameters are frozen at build time from
ColumnSchema.metadata, never inferred from the incoming batch. The chain of
builders is configured on PipelineBuilder; each builder is itself stype-aware
and returns None for columns the step does not apply to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from data.batch import DataBatch
from data.base import ColumnStype, MetadataKey
from data.schema import ColumnSchema, DatabaseSchema

from .base import ColumnEncoder, ColumnPipeline, Identity, Operator


TimestampStrategy = Literal["epoch_s", "epoch_ms", "epoch_ns"]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedFeatures:
    """Per-column encoded arrays — node features in GNN terms.

    Keyed by (table, column); values are framework-agnostic numpy arrays.
    """

    features: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Terminal encoders (Arrow -> ndarray)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalEncoder:
    """Map categorical values to frozen integer codes via list position."""

    cardinality: List[str]
    _value_set: pa.Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_value_set", pa.array(self.cardinality))

    def encode(self, arr: pa.Array) -> np.ndarray:
        codes = pc.index_in(arr, value_set=self._value_set) # type: ignore
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
        return pc.cast(arr, pa.timestamp(unit)).cast(pa.int64()).to_numpy(
            zero_copy_only=False
        )


# ---------------------------------------------------------------------------
# Encoder builder (ColumnSchema -> ColumnEncoder | None)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Operator builders (ColumnSchema -> Operator)
# ---------------------------------------------------------------------------


class OperatorBuilder(Protocol):
    """Produce an Operator configured for one column.

    Returns ``None`` when the step does not apply to the column;
    ``PipelineBuilder`` drops those so pipeline assembly stays branch-free.
    """

    def __call__(self, col: ColumnSchema) -> Optional[Operator]: ...


# ---- Null-fill -----------------------------------------------------------


class NullFillBuilder:
    """Build a null-fill operator from column metadata.

    Strategies self-register via ``@NullFillBuilder.register("<name>")``.
    Adding a new strategy is one file + one decorator; no builder edits.
    """

    _registry: Dict[str, Type[Operator]] = {"default": Identity}

    _stype_nullfill: Dict[ColumnStype, str] = {
        ColumnStype.NUMERICAL: "constant",
        ColumnStype.CATEGORICAL: "forward",
        ColumnStype.TIMESTAMP: "default",
    }

    @classmethod
    def register(cls, name: str) -> Callable[[Type[Operator]], Type[Operator]]:
        def decorator(op_cls: Type[Operator]) -> Type[Operator]:
            if name in cls._registry:
                raise ValueError(
                    f"null-fill strategy already registered: {name!r}"
                )
            cls._registry[name] = op_cls
            return op_cls

        return decorator

    def __call__(self, col: ColumnSchema) -> Optional[Operator]:
        stype = col.metadata.get(MetadataKey.STYPE)
        if stype is None:
            return None
        strategy = self._stype_nullfill.get(stype)
        if strategy is None:
            return None
        return self._registry[strategy]()


@NullFillBuilder.register("constant")
@dataclass(frozen=True)
class NullFillConstant:
    """Replace nulls with a fixed scalar (mean/median resolved offline)."""

    value: Any

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null(arr, pa.scalar(self.value, type=arr.type))


@NullFillBuilder.register("forward")
@dataclass(frozen=True)
class NullFillForward:
    """Carry the last observed value forward across nulls."""

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null_forward(arr) # type: ignore


@NullFillBuilder.register("backward")
@dataclass(frozen=True)
class NullFillBackward:
    """Carry the next observed value backward across nulls."""

    def apply(self, arr: pa.Array) -> pa.Array:
        return pc.fill_null_backward(arr) # type: ignore


# ---------------------------------------------------------------------------
# Pipeline builder (ColumnSchema -> ColumnPipeline | None)
# ---------------------------------------------------------------------------


@dataclass
class PipelineBuilder:
    """Build a ``ColumnPipeline`` for a column by running each builder in turn.

    Builders are stype-aware on their own (``NullFillBuilder``,
    ``EncoderBuilder``); they return ``None`` for columns the step does not
    apply to. Operators (Arrow -> Arrow) and the terminal encoder
    (Arrow -> ndarray) are kept as separate fields so the contract — exactly
    one encoder, always last — is enforced at construction time, not at
    runtime.
    """

    operator_builders: Tuple[OperatorBuilder, ...] = (NullFillBuilder(),)
    encoder_builder: EncoderBuilder = field(default_factory=EncoderBuilder)

    def build(self, col: ColumnSchema) -> Optional[ColumnPipeline]:
        encoder = self.encoder_builder(col)
        if encoder is None:
            return None
        operators = tuple(
            op for b in self.operator_builders if (op := b(col)) is not None
        )
        return ColumnPipeline(operators=operators, encoder=encoder)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


@dataclass
class FeatureConverter:
    """Apply schema-backed pipelines to an Arrow DataBatch.

    Iteration is driven by the batch: the RecordBatch reflects any projection
    pushed down by the engine, so only columns actually present are encoded.
    The DatabaseSchema is a lookup reference — for each Arrow column we
    resolve the matching ColumnSchema and build a pipeline on first use.

    Pipeline construction is delegated to ``pipeline_builder``; each builder
    in its chain is itself stype-aware and returns ``None`` to skip. Pipelines
    are cached per (table, column) so cardinality-list / mean-std / operator
    construction is paid once per column across the converter's lifetime.
    """

    schema: DatabaseSchema
    pipeline_builder: PipelineBuilder = field(default_factory=PipelineBuilder)
    _cache: Dict[Tuple[str, str], ColumnPipeline] = field(
        default_factory=dict, init=False, repr=False
    )

    def apply(self, batch: DataBatch) -> EncodedFeatures:
        features: Dict[Tuple[str, str], np.ndarray] = {}
        for table_name, rb in batch.tables.items():
            table_schema = self.schema.tables.get(table_name)
            if table_schema is None:
                continue
            for col_name in rb.schema.names:
                col = table_schema.columns.get(col_name)
                if col is None:
                    continue
                pipeline = self._get_pipeline(table_name, col_name, col)
                if pipeline is None:
                    continue
                features[(table_name, col_name)] = pipeline.apply(
                    rb.column(col_name)
                )
        return EncodedFeatures(features=features)

    def _get_pipeline(
        self, table_name: str, col_name: str, col: ColumnSchema
    ) -> Optional[ColumnPipeline]:
        key = (table_name, col_name)
        pipeline = self._cache.get(key)
        if pipeline is not None:
            return pipeline
        pipeline = self.pipeline_builder.build(col)
        if pipeline is not None:
            self._cache[key] = pipeline
        return pipeline
