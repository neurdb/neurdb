"""Arrow -> numpy conversion layer sitting between DataBatch and the model.

Handles per-column encoding (categorical tokenization, numeric normalization,
timestamp casting) and relational edge building from the schema's FK graph.
Output is framework-agnostic numpy; the trainer wraps it in torch/tf/jax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .batch import DataBatch
from .base import ColumnStype, MetadataKey
from .schema import (
    ColumnRole,
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
)


UnknownPolicy = Literal["strict", "unk_token"]
TimestampStrategy = Literal["epoch_s", "epoch_ms", "epoch_ns"]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInput:
    """Converter output. Everything is np.ndarray; no torch/tf dependency."""

    features: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    targets: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    timestamps: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    # Relational edges keyed by RelationshipSchema.name.
    # Value is (src_row_idx, tgt_row_idx) into the respective record batches.
    edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    row_counts: Dict[str, int] = field(default_factory=dict)
    anchor_table: str = ""


# ---------------------------------------------------------------------------
# Per-column encoders
# ---------------------------------------------------------------------------


class ColumnEncoder(Protocol):
    def encode(self, arr: pa.Array) -> np.ndarray: ...


@dataclass(frozen=True)
class CategoricalEncoder:
    """Map categorical values to frozen integer codes via list position."""

    cardinality: List[str]
    _value_set: pa.Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_value_set", pa.array(self.cardinality))

    def encode(self, arr: pa.Array) -> np.ndarray:
        codes = pc.index_in(arr, value_set=self._value_set)
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
# Converter
# ---------------------------------------------------------------------------


@dataclass
class FeatureConverter:
    """Apply schema-backed encoders to an Arrow DataBatch.

    Iteration is driven by the batch: the RecordBatch reflects any projection
    pushed down by the engine, so only columns actually present are encoded.
    The DatabaseSchema is a lookup reference — for each Arrow column we
    resolve the matching ColumnSchema and build an encoder on first use.

    Encoders are cached per (table, column) so cardinality-list / mean-std
    construction is paid once per column across the converter's lifetime.
    """

    schema: DatabaseSchema
    unknown: UnknownPolicy = "strict"
    timestamp_strategy: TimestampStrategy = "epoch_s"
    _cache: Dict[Tuple[str, str], ColumnEncoder] = field(
        default_factory=dict, init=False, repr=False
    )

    # ---- application ---------------------------------------------------

    def apply(self, batch: DataBatch) -> ModelInput:
        features: Dict[Tuple[str, str], np.ndarray] = {}
        targets: Dict[Tuple[str, str], np.ndarray] = {}
        timestamps: Dict[Tuple[str, str], np.ndarray] = {}
        row_counts: Dict[str, int] = {
            name: rb.num_rows for name, rb in batch.tables.items()
        }

        for table_name, rb in batch.tables.items():
            table_schema = self.schema.tables.get(table_name)
            if table_schema is None:
                continue
            for col_name in rb.schema.names:
                col = table_schema.columns.get(col_name)
                if col is None:
                    continue
                encoder = self._get_encoder(table_name, col_name, col)
                if encoder is None:
                    continue
                values = encoder.encode(rb.column(col_name))
                bucket = _bucket_for_role(col.role, features, targets, timestamps)
                if bucket is not None:
                    bucket[(table_name, col_name)] = values

        edges = self._build_edges(batch)

        return ModelInput(
            features=features,
            targets=targets,
            timestamps=timestamps,
            edges=edges,
            row_counts=row_counts,
            anchor_table=batch.anchor_table,
        )

    # ---- introspection used by the model -------------------------------

    def vocab_sizes(self) -> Dict[Tuple[str, str], int]:
        """Embedding-table sizes from the schema, independent of any batch."""
        out: Dict[Tuple[str, str], int] = {}
        for table_name, col_name, col in self.schema.iter_columns():
            if col.metadata.get(MetadataKey.STYPE) != ColumnStype.CATEGORICAL:
                continue
            cardinality = (col.metadata.get(MetadataKey.STATS) or {}).get(MetadataKey.CARDINALITY)
            if cardinality:
                out[(table_name, col_name)] = len(cardinality)
        return out

    # ---- lazy encoder construction -------------------------------------

    def _get_encoder(
        self, table_name: str, col_name: str, col: ColumnSchema
    ) -> Optional[ColumnEncoder]:
        key = (table_name, col_name)
        encoder = self._cache.get(key)
        if encoder is not None:
            return encoder
        encoder = self._build_encoder(col)
        if encoder is not None:
            self._cache[key] = encoder
        return encoder

    def _build_encoder(self, col: ColumnSchema) -> Optional[ColumnEncoder]:
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

    # ---- edge building -------------------------------------------------

    def _build_edges(
        self, batch: DataBatch
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for rel in self.schema.relationships:
            if rel.source_table not in batch.tables:
                continue
            if rel.target_table not in batch.tables:
                continue
            edges[rel.name] = _join_rows(batch, rel)
        return edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bucket_for_role(
    role: ColumnRole,
    features: Dict[Tuple[str, str], np.ndarray],
    targets: Dict[Tuple[str, str], np.ndarray],
    timestamps: Dict[Tuple[str, str], np.ndarray],
) -> Optional[Dict[Tuple[str, str], np.ndarray]]:
    if role is ColumnRole.FEATURE:
        return features
    if role is ColumnRole.TARGET:
        return targets
    if role is ColumnRole.TIMESTAMP:
        return timestamps
    return None


def _join_rows(
    batch: DataBatch, rel: RelationshipSchema
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (src_row_idx, tgt_row_idx) implementing rel as an inner join."""
    src_rb = batch.tables[rel.source_table]
    tgt_rb = batch.tables[rel.target_table]

    src_keys = _composite_key(src_rb, rel.source_columns)
    tgt_keys = _composite_key(tgt_rb, rel.target_columns)

    tgt_index: Dict[Tuple[Any, ...], int] = {k: i for i, k in enumerate(tgt_keys)}

    src_idx: List[int] = []
    tgt_idx: List[int] = []
    for i, k in enumerate(src_keys):
        j = tgt_index.get(k)
        if j is None:
            continue
        src_idx.append(i)
        tgt_idx.append(j)
    return np.asarray(src_idx, dtype=np.int64), np.asarray(tgt_idx, dtype=np.int64)


def _composite_key(rb: pa.RecordBatch, columns: List[str]) -> List[Tuple[Any, ...]]:
    cols = [rb.column(c).to_pylist() for c in columns]
    return list(zip(*cols))
