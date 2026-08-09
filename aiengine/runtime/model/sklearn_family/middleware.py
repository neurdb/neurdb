"""SklearnMiddleware: EncodedBatch stream -> MatrixInput.

The sklearn family's packager. It materializes the job's cached stream
into one design matrix (FIT_ONCE models want all rows at once) following
the anchor rule: only the *target table's* columns are packed — non-anchor
tables in a multi-table view are ignored (the trainer warns).

``MatrixInput`` is self-describing: ``columns[j]`` describes ``X[:, j]``.
Types are DECLARED, never estimated — ``MatrixColumn.stype`` is the
engine's declaration from the DatabaseSchema, carried verbatim through the
view and the spec; the runtime only moves the label around as column
positions change. Every pluggable step that reshapes the matrix must
re-emit descriptors to match (that is the plugin contract), so positional
conventions never go stale.

Boxes build their estimator FROM the descriptors
(``SklearnModel.build_estimator(columns)``) — e.g. logreg one-hots the
categorical positions — so estimator construction is always consistent
with the actual matrix, whatever steps ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np
from data.base import ColumnStype, MetadataKey
from pipeline.input import EncodedBatch

from ..base import ModelSpec


@dataclass(frozen=True)
class MatrixColumn:
    """Descriptor for one matrix column: provenance + declared type."""

    key: Tuple[str, str]  # (table, column) it came from
    stype: ColumnStype  # the DatabaseSchema's declaration
    n_categories: Optional[int] = None  # incl. reserved unknown code


@dataclass(frozen=True)
class MatrixInput:
    """The sklearn family's model input: a self-describing matrix."""

    X: np.ndarray  # (n, d); column j described by columns[j]
    y: np.ndarray  # (n,) encoded target
    columns: Tuple[MatrixColumn, ...]

    @property
    def categorical_idx(self) -> Tuple[int, ...]:
        return tuple(
            j for j, c in enumerate(self.columns) if c.stype is ColumnStype.CATEGORICAL
        )

    @property
    def numeric_idx(self) -> Tuple[int, ...]:
        return tuple(
            j
            for j, c in enumerate(self.columns)
            if c.stype is not ColumnStype.CATEGORICAL
        )


class MatrixStep(Protocol):
    """Pluggable matrix transform. MUST keep descriptors aligned with X."""

    def __call__(self, matrix: MatrixInput) -> MatrixInput: ...


class SklearnMiddleware:
    def __init__(self, spec: ModelSpec, steps: Tuple[MatrixStep, ...] = ()) -> None:
        self._spec = spec
        self._steps = steps
        anchor = spec.target[0]
        self._keys = tuple(k for k in spec.feature_order if k[0] == anchor)
        n_categories = {f.key: f.n_embeddings for f in spec.categorical_features}
        self._columns = tuple(
            MatrixColumn(
                key=key,
                stype=ColumnStype(
                    spec.feature_schema.get_column(*key).metadata[MetadataKey.STYPE]
                ),
                n_categories=n_categories.get(key),
            )
            for key in self._keys
        )

    def __call__(self, stream: Iterable[EncodedBatch]) -> MatrixInput:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for encoded in stream:
            features = encoded.features.features
            try:
                cols = [
                    features[key].astype(np.float32, copy=False) for key in self._keys
                ]
                y = features[self._spec.target]
            except KeyError as exc:
                raise KeyError(
                    f"batch is missing encoded column {exc.args[0]}; the "
                    "engine's projection must include every feature the spec "
                    "was derived from"
                ) from exc
            xs.append(np.column_stack(cols) if cols else np.empty((len(y), 0)))
            ys.append(y)
        if not xs:
            raise ValueError("empty data stream: nothing to pack")

        matrix = MatrixInput(
            X=np.vstack(xs), y=np.concatenate(ys), columns=self._columns
        )
        for step in self._steps:
            matrix = step(matrix)
        return matrix
