"""Type-aware feature preprocessing for tabular AI operators (e.g. TabPFN).

Given a (train, test) pair of pandas DataFrames produced database-natively
(e.g. the ``w_task_<h>`` tables from the avito AdCTR workload), this module:

1. Infers a semantic type (stype) per column: ``numerical`` / ``categorical`` /
   ``timestamp`` / ``text`` / ``drop``. Inference combines pandas dtype, column
   name keywords (same spirit as NeurIDA ``utils/preprocess.py``), id/key hints,
   and a low-cardinality (sparsity) rule. Explicit ``col_to_stype`` hints always win.
2. Transforms columns into a dense numeric matrix the model can consume:
   - numerical  -> median-imputed float
   - categorical -> most-frequent-imputed + label-encoded ints (unseen -> -1)
   - timestamp  -> expanded into numeric parts (year/month/day/dow/hour/epoch)
   - text/drop  -> dropped
   It is fit on train and applied to both train and test (no leakage), and
   reports the categorical feature indices that models like TabPFN want.

The implementation is dependency-light (pandas / numpy / scikit-learn only) so it
can run inside the runtime without pulling torch_frame / relbench.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Semantic type tags.
NUMERICAL = "numerical"
CATEGORICAL = "categorical"
TIMESTAMP = "timestamp"
TEXT = "text"
DROP = "drop"

# Column-name keyword rules (mirrors NeurIDA utils/preprocess.py).
NUMERICAL_KEYWORDS = {
    "count", "num", "amount", "total", "length", "height", "value", "rate",
    "number", "score", "size", "price", "percent", "ratio", "volume", "avg",
    "max", "min", "age", "ctr", "impr", "impressions", "click", "clicks",
}
CATEGORICAL_KEYWORDS = {
    "type", "category", "class", "label", "status", "code", "id", "guid",
    "region", "zone", "flag", "is", "has", "mode", "city", "context", "level",
    "parent",
}
TEXT_KEYWORDS = {
    "description", "comments", "content", "name", "review", "message", "note",
    "query", "summary", "title", "text",
}

_TS_PARTS = ("year", "month", "day", "dayofweek", "hour")


def _tokenize(identifier: str) -> List[str]:
    """Split snake_case / kebab-case / CamelCase into lowercase tokens."""
    parts = re.split(r"[_\-]", identifier)
    tokens: List[str] = []
    for part in parts:
        # split CamelCase and trailing digits (e.g. feature10 -> feature, 10)
        sub = re.findall(r"[A-Za-z][a-z]*|\d+", part)
        tokens.extend(sub if sub else [part])
    return [t.lower() for t in tokens if t]


def infer_column_stypes(
    df: pd.DataFrame,
    *,
    hints: Optional[Dict[str, str]] = None,
    id_cols: Sequence[str] = (),
    low_card_ratio: float = 0.01,
    text_card_ratio: float = 0.5,
) -> Dict[str, str]:
    """Propose a semantic type for each column of ``df``.

    Args:
        hints: explicit column -> stype mapping; takes precedence over inference.
        id_cols: columns to drop outright (entity/primary keys, not features).
        low_card_ratio: numeric/text columns with unique/non-null ratio below this
            are demoted to categorical (sparse codes).
        text_card_ratio: object columns with unique/non-null ratio above this are
            treated as free text (otherwise categorical).
    """
    hints = dict(hints or {})
    id_cols = set(id_cols)
    out: Dict[str, str] = {}

    for col in df.columns:
        if col in hints:
            out[col] = hints[col]
            continue
        if col in id_cols:
            out[col] = DROP
            continue

        s = df[col]
        tokens = set(_tokenize(col))
        n_nonnull = int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))

        # base guess from dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            base = TIMESTAMP
        elif pd.api.types.is_bool_dtype(s):
            base = CATEGORICAL
        elif pd.api.types.is_numeric_dtype(s):
            base = NUMERICAL
        else:
            ratio = (n_unique / n_nonnull) if n_nonnull else 0.0
            base = TEXT if ratio > text_card_ratio else CATEGORICAL

        # name-keyword overrides (do not override a detected timestamp)
        if base != TIMESTAMP:
            if tokens & TEXT_KEYWORDS:
                base = TEXT
            elif (tokens & CATEGORICAL_KEYWORDS) and not (tokens & NUMERICAL_KEYWORDS):
                base = CATEGORICAL
            elif (tokens & NUMERICAL_KEYWORDS) and pd.api.types.is_numeric_dtype(s):
                base = NUMERICAL

        # sparsity rule: very low cardinality numeric/text -> categorical
        if base in (NUMERICAL, TEXT) and n_nonnull > 0:
            if (n_unique / n_nonnull) < low_card_ratio:
                base = CATEGORICAL

        out[col] = base

    return out


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    categorical_indices: List[int]
    col_to_stype: Dict[str, str]


class TabularPreprocessor:
    """Fit on train, transform train/test into numeric arrays for a tabular model."""

    def __init__(
        self,
        col_to_stype: Optional[Dict[str, str]] = None,
        *,
        id_cols: Sequence[str] = (),
        num_impute: str = "median",
        cat_impute: str = "most_frequent",
        expand_timestamp: bool = True,
        low_card_ratio: float = 0.01,
    ):
        self.col_to_stype = col_to_stype
        self.id_cols = list(id_cols)
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.expand_timestamp = expand_timestamp
        self.low_card_ratio = low_card_ratio

        # learned during fit
        self._numerical_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._timestamp_cols: List[str] = []
        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._cat_maps: Dict[str, Dict[object, int]] = {}
        self._feature_names: List[str] = []
        self._categorical_indices: List[int] = []
        self._fitted = False

    # -- public API -----------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "TabularPreprocessor":
        feat_df = train_df.drop(columns=[target_col])

        stypes = self.col_to_stype or infer_column_stypes(
            feat_df, id_cols=self.id_cols, low_card_ratio=self.low_card_ratio
        )
        self.col_to_stype = stypes

        self._numerical_cols = [c for c in feat_df.columns if stypes.get(c) == NUMERICAL]
        self._categorical_cols = [c for c in feat_df.columns if stypes.get(c) == CATEGORICAL]
        self._timestamp_cols = (
            [c for c in feat_df.columns if stypes.get(c) == TIMESTAMP]
            if self.expand_timestamp
            else []
        )

        # numerical imputer (fit on numerical + expanded timestamp parts)
        num_block = self._numerical_frame(feat_df)
        if not num_block.empty:
            # keep_empty_features: never silently drop all-NaN columns, so the
            # feature layout stays fixed between fit and transform.
            self._num_imputer = SimpleImputer(
                strategy=self.num_impute, keep_empty_features=True
            )
            self._num_imputer.fit(num_block)

        # categorical imputer + label maps
        if self._categorical_cols:
            cat_raw = feat_df[self._categorical_cols].astype(object)
            self._cat_imputer = SimpleImputer(
                strategy=self.cat_impute, keep_empty_features=True
            )
            cat_imp = pd.DataFrame(
                self._cat_imputer.fit_transform(cat_raw),
                columns=self._categorical_cols,
                index=cat_raw.index,
            )
            for col in self._categorical_cols:
                classes = pd.Index(cat_imp[col].unique())
                self._cat_maps[col] = {v: i for i, v in enumerate(classes)}

        # feature layout: numerical block first, then categorical
        num_names = list(num_block.columns)
        self._feature_names = num_names + list(self._categorical_cols)
        self._categorical_indices = list(
            range(len(num_names), len(num_names) + len(self._categorical_cols))
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor.transform called before fit")

        blocks: List[np.ndarray] = []

        num_block = self._numerical_frame(df)
        if not num_block.empty:
            if self._num_imputer is not None:
                num_arr = self._num_imputer.transform(num_block)
            else:
                num_arr = num_block.to_numpy(dtype=float)
            blocks.append(np.asarray(num_arr, dtype=float))

        if self._categorical_cols:
            cat_raw = df[self._categorical_cols].astype(object)
            cat_imp = pd.DataFrame(
                self._cat_imputer.transform(cat_raw),
                columns=self._categorical_cols,
                index=cat_raw.index,
            )
            cat_cols = []
            for col in self._categorical_cols:
                mapping = self._cat_maps[col]
                codes = cat_imp[col].map(mapping)
                codes = codes.fillna(-1).astype(float)  # unseen category -> -1
                cat_cols.append(codes.to_numpy())
            blocks.append(np.column_stack(cat_cols))

        if not blocks:
            return np.empty((len(df), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
    ) -> PreprocessResult:
        self.fit(train_df, target_col)
        return PreprocessResult(
            X_train=self.transform(train_df.drop(columns=[target_col])),
            X_test=self.transform(test_df.drop(columns=[target_col])),
            y_train=train_df[target_col].to_numpy(),
            y_test=test_df[target_col].to_numpy(),
            feature_names=list(self._feature_names),
            categorical_indices=list(self._categorical_indices),
            col_to_stype=dict(self.col_to_stype or {}),
        )

    # -- helpers --------------------------------------------------------------

    def _numerical_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numerical columns plus expanded timestamp parts as one numeric frame."""
        cols = {}
        for c in self._numerical_cols:
            cols[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self._timestamp_cols:
            ts = pd.to_datetime(df[c], errors="coerce")
            for part in _TS_PARTS:
                cols[f"{c}__{part}"] = getattr(ts.dt, part).astype("float64")
            # seconds since epoch; NaT -> NaN (handled later by the imputer)
            cols[f"{c}__epoch"] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
        if not cols:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(cols, index=df.index)
