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

Concrete steps live in sibling modules: ``pipeline.feature.encoder``
(terminal encoders + EncoderBuilder) and ``pipeline.feature.nullfill``
(null-fill operators).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import numpy as np
from data.base import ColumnRole
from data.batch import DataBatch
from data.schema import ColumnSchema, FeatureSchema

from .base import ColumnPipeline, EncodeError, OperatorBuilder
from .encoder import EncoderBuilder
from .nullfill import NullFillBuilder

logger = logging.getLogger(__name__)

# Roles whose columns are meant to reach the model — dropping one of these
# is a schema bug worth shouting about, not a routine projection skip.
_MODEL_ROLES = (ColumnRole.FEATURE, ColumnRole.TARGET)


@dataclass(frozen=True)
class EncodedFeatures:
    """Per-column encoded arrays — node features in GNN terms.

    Keyed by (table, column); values are framework-agnostic numpy arrays.
    """

    features: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)


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


@dataclass
class FeatureConverter:
    """Apply schema-backed pipelines to an Arrow DataBatch.

    Iteration is driven by the batch: the RecordBatch reflects any projection
    pushed down by the engine, so only columns actually present are encoded.
    The ``FeatureSchema`` — a slim view derived at job setup by the
    ViewBuilder — is the lookup reference: for each Arrow column we resolve
    the matching ColumnSchema and build a pipeline on first use. Tables and
    columns outside the view (e.g. squeezed junctions) are simply absent
    from it and skip without any runtime filtering.

    Pipeline construction is delegated to ``pipeline_builder``; each builder
    in its chain is itself stype-aware and returns ``None`` to skip. Pipelines
    are cached per (table, column) so cardinality-list / mean-std / operator
    construction is paid once per column across the converter's lifetime.

    Columns that yield no pipeline are skipped — silently for key/metadata
    roles, with a one-time warning for FEATURE/TARGET roles, where a missing
    pipeline means broken schema metadata, not intent.
    """

    schema: FeatureSchema
    pipeline_builder: PipelineBuilder = field(default_factory=PipelineBuilder)
    _cache: Dict[Tuple[str, str], ColumnPipeline] = field(
        default_factory=dict, init=False, repr=False
    )
    _warned: Set[Tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    def apply(self, batch: DataBatch) -> EncodedFeatures:
        features: Dict[Tuple[str, str], np.ndarray] = {}
        for table_name, rb in batch.tables.items():
            table_columns = self.schema.tables.get(table_name)
            if table_columns is None:
                continue
            for col_name in rb.schema.names:
                col = table_columns.get(col_name)
                if col is None:
                    continue
                pipeline = self._get_pipeline(table_name, col_name, col)
                if pipeline is None:
                    self._warn_dropped(table_name, col_name, col)
                    continue
                try:
                    features[(table_name, col_name)] = pipeline.apply(
                        rb.column(col_name)
                    )
                except Exception as exc:
                    raise EncodeError(
                        f"failed to encode column {table_name}.{col_name}: {exc}"
                    ) from exc
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

    def _warn_dropped(self, table_name: str, col_name: str, col: ColumnSchema) -> None:
        key = (table_name, col_name)
        if col.role not in _MODEL_ROLES or key in self._warned:
            return
        self._warned.add(key)
        logger.warning(
            "dropping %s column %s.%s: no pipeline could be built from its "
            "schema metadata (missing/unknown stype or required stats)",
            col.role.value,
            table_name,
            col_name,
        )
