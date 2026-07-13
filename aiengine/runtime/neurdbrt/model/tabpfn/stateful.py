"""Stateful, in-context TabPFN for the streaming engine path.

TabPFN "trains" by conditioning on a labelled context and predicts in a single
forward pass. The DB engine streams data in two phases:

1. **context phase** (``stage == "train"``): the whole labelled context arrives
   in batches -> :meth:`StatefulTabPFN.fit_context` fits the type-aware
   preprocessor and caches the context inside TabPFN *once*.
2. **predict phase** (``stage == "inference"``): test rows arrive in batches ->
   :meth:`StatefulTabPFN.predict_batch` transforms each batch with the *already
   fitted* preprocessor and predicts against the cached context.

This object is what gets cached per ``model_id`` on the engine (see
``session_store``) so the context survives across the streamed test batches.

Kept base-free (no ``neurdbrt.model.base`` / websocket / cache imports) so it can
be unit-tested in a lean TabPFN environment.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # normal package import
    from .model import BINARY, REGRESSION, TabPFNPredictor
    from .preprocess import (
        CATEGORICAL,
        DROP,
        NUMERICAL,
        TEXT,
        TIMESTAMP,
        TabularPreprocessor,
    )
except ImportError:  # loaded standalone (dir on sys.path), e.g. unit test
    from model import BINARY, REGRESSION, TabPFNPredictor  # type: ignore
    from preprocess import (  # type: ignore
        CATEGORICAL,
        DROP,
        NUMERICAL,
        TEXT,
        TIMESTAMP,
        TabularPreprocessor,
    )


# Postgres type name (lower-cased, as reported by format_type) -> stype hint.
#
# The typed wire format sends every value as a *string* (tab-separated), so the
# data-driven ``infer_column_stypes`` cannot tell a numeric column from text:
# a float CTR like "0.0123" looks like a high-cardinality text/categorical
# column. We therefore pin the columns whose semantic type the DB already knows
# and the engine cannot recover from strings: numerics (so they are coerced back
# to float) and timestamps (so they are expanded into numeric parts). Booleans
# are pinned categorical; ``text``/unknown types are left to data inference.
_PG_TYPE_TO_STYPE: Dict[str, str] = {
    # timestamps / dates
    "timestamp": TIMESTAMP,
    "timestamp without time zone": TIMESTAMP,
    "timestamp with time zone": TIMESTAMP,
    "timestamptz": TIMESTAMP,
    "date": TIMESTAMP,
    # integers
    "smallint": NUMERICAL,
    "integer": NUMERICAL,
    "bigint": NUMERICAL,
    "int2": NUMERICAL,
    "int4": NUMERICAL,
    "int8": NUMERICAL,
    # reals / fixed point
    "real": NUMERICAL,
    "double precision": NUMERICAL,
    "float4": NUMERICAL,
    "float8": NUMERICAL,
    "numeric": NUMERICAL,
    "decimal": NUMERICAL,
    "money": NUMERICAL,
    # booleans
    "boolean": CATEGORICAL,
    "bool": CATEGORICAL,
}


def pg_types_to_hints(
    feature_names: Sequence[str], pg_types: Sequence[str]
) -> Dict[str, str]:
    """Map ``(name, pg_type)`` pairs to soft stype hints for the preprocessor.

    Unknown/ambiguous pg types are simply omitted (left to data inference).
    """
    hints: Dict[str, str] = {}
    for name, pg in zip(feature_names, pg_types):
        # normalise e.g. "numeric(10,2)" -> "numeric", "timestamp(6)" -> "timestamp"
        norm = re.sub(r"\s*\([^)]*\)", "", str(pg).strip().lower())
        stype = _PG_TYPE_TO_STYPE.get(norm)
        if stype is not None:
            hints[name] = stype
    return hints


def infer_task_type(n_class: int) -> str:
    """Map the DB-side class count to a TabPFN task type.

    ``PREDICT VALUE`` sends ``n_class <= 0`` (regression); ``PREDICT CLASS``
    sends the number of classes (binary => 2).
    """
    return REGRESSION if (n_class is None or n_class <= 0) else BINARY


class StatefulTabPFN:
    def __init__(
        self,
        target_col: str,
        task_type: str = REGRESSION,
        *,
        feature_names: Optional[Sequence[str]] = None,
        stype_hints: Optional[Dict[str, str]] = None,
        id_cols: Sequence[str] = (),
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        max_train_samples: int = 10_000,
        batch_size: int = 4096,
    ):
        self.target_col = target_col
        self.task_type = task_type
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.device = device
        self.model_path = model_path
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size

        self._pre = TabularPreprocessor(stype_hints=stype_hints, id_cols=id_cols)
        self._model: Optional[TabPFNPredictor] = None
        self._fitted = False
        self.n_context = 0

    # -- context phase --------------------------------------------------------

    def fit_context(self, feat_df: pd.DataFrame, y: Sequence) -> "StatefulTabPFN":
        """Fit the preprocessor + cache the labelled context inside TabPFN."""
        y_series = self._coerce_target(pd.Series(list(y)))
        # the preprocessor fits on a frame that *includes* the target column.
        context = feat_df.copy()
        context[self.target_col] = y_series.to_numpy()

        self._pre.fit(context, self.target_col)
        X = self._pre.transform(feat_df)
        y_arr = y_series.to_numpy()

        self._model = TabPFNPredictor(
            task_type=self.task_type,
            device=self.device,
            model_path=self.model_path,
            categorical_indices=self._pre._categorical_indices,
            max_train_samples=self.max_train_samples,
            batch_size=self.batch_size,
        )
        self._model.fit(X, y_arr)
        self._fitted = True
        self.n_context = int(X.shape[0])
        return self

    # -- predict phase --------------------------------------------------------

    def predict_batch(self, feat_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("StatefulTabPFN.predict_batch called before fit_context")
        X = self._pre.transform(feat_df)
        return self._model.predict(X)

    # -- helpers --------------------------------------------------------------

    def _coerce_target(self, y: pd.Series) -> pd.Series:
        if self.task_type == REGRESSION:
            return pd.to_numeric(y, errors="coerce").astype(float)
        # classification: numeric labels stay numeric, otherwise label-encode
        num = pd.to_numeric(y, errors="coerce")
        if num.notna().all():
            return num.astype(int)
        return y.astype("category").cat.codes.astype(int)

    @property
    def col_to_stype(self) -> Dict[str, str]:
        return dict(self._pre.col_to_stype or {})

    @property
    def feature_layout(self) -> List[str]:
        return list(self._pre._feature_names)
