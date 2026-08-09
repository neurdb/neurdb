"""Middleware bridging EncodedFeatures to the global token:value format.

The pipeline's encoders emit *column-local* values: categorical codes are
positions in each column's own cardinality list, so code ``2`` from
``users.country`` and code ``2`` from ``users.city`` are different things
that collide numerically. Embedding-consuming models (tabular torch boxes,
relational/GNN boxes) need *globally unique* tokens over one shared
vocabulary. ``FeatureTokenizer`` is that bridge, and it is deliberately
framework-free (numpy in, numpy out) — torch is just one consumer.

Every cell becomes ``token_id:value``:

* numerical column   -> one global token per column, value = the encoded
  number (embedding = value * E[token]);
* categorical column -> a disjoint token block of ``n_embeddings`` ids
  (reserved unknown code included), token = block offset + local code,
  value = 1.0.

The vocabulary layout is deterministic from the ``ModelSpec``: numeric
column tokens first ``[0, n_numeric)``, then one contiguous block per
categorical column, both in the spec's canonical order. Output is grouped
per table — SINGLE_TABLE models use their one table; relational models
get one (tokens, values) pair per node table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pipeline.feature.converter import EncodedFeatures

from .base import ModelSpec


@dataclass(frozen=True)
class _Field:
    key: Tuple[str, str]
    offset: int
    is_categorical: bool


class FeatureTokenizer:
    """EncodedFeatures -> {table: (tokens (n,F) int64, values (n,F) float32)}."""

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        n_numeric = len(spec.numeric_features)

        fields: Dict[str, List[_Field]] = {}
        for j, key in enumerate(spec.numeric_features):
            fields.setdefault(key[0], []).append(
                _Field(key=key, offset=j, is_categorical=False)
            )
        offset = n_numeric
        for feature in spec.categorical_features:
            fields.setdefault(feature.key[0], []).append(
                _Field(key=feature.key, offset=offset, is_categorical=True)
            )
            offset += feature.n_embeddings

        self._fields = fields
        self.vocab_size = offset
        self.n_fields = n_numeric + len(spec.categorical_features)

    def fields_of(self, table: str) -> int:
        """Number of fields a table contributes."""
        return len(self._fields.get(table, []))

    def transform(
        self, features: EncodedFeatures
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for table, table_fields in self._fields.items():
            token_cols: List[np.ndarray] = []
            value_cols: List[np.ndarray] = []
            for field in table_fields:
                try:
                    arr = features.features[field.key]
                except KeyError as exc:
                    raise KeyError(
                        f"batch is missing encoded column {exc.args[0]}; the "
                        "engine's projection must include every feature the "
                        "spec was derived from"
                    ) from exc
                if field.is_categorical:
                    token_cols.append(field.offset + arr.astype(np.int64, copy=False))
                    value_cols.append(np.ones(len(arr), dtype=np.float32))
                else:
                    token_cols.append(np.full(len(arr), field.offset, dtype=np.int64))
                    value_cols.append(arr.astype(np.float32, copy=False))
            out[table] = (
                np.column_stack(token_cols),
                np.column_stack(value_cols),
            )
        return out
