"""Base-free orchestration: type-aware preprocess + TabPFN predict + metrics.

This module deliberately avoids importing ``neurdbrt.model.base`` (and therefore
the websocket/cache server stack), so it can be imported standalone -- e.g. by
the unit test in a lean TabPFN environment -- as well as from the runtime
``builder``. Imports of the sibling leaf modules fall back to a flat import so
the file works both inside the package and when its directory is on ``sys.path``.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:  # normal package import
    from .preprocess import TabularPreprocessor
    from .model import BINARY, REGRESSION, TabPFNPredictor
except ImportError:  # loaded standalone (dir on sys.path), e.g. unit test
    from preprocess import TabularPreprocessor  # type: ignore
    from model import BINARY, REGRESSION, TabPFNPredictor  # type: ignore


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    out: Dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")
    out["accuracy"] = float(accuracy_score(y_true, (y_score >= 0.5).astype(int)))
    return out


def run_tabular_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: str = REGRESSION,
    *,
    col_to_stype: Optional[Dict[str, str]] = None,
    id_cols: Sequence[str] = (),
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    max_train_samples: int = 10_000,
    batch_size: int = 4096,
) -> Dict:
    """Type-aware preprocess + TabPFN predict; returns predictions, metrics, timings."""
    t0 = time.time()
    pre = TabularPreprocessor(col_to_stype=col_to_stype, id_cols=id_cols)
    data = pre.fit_transform(train_df, test_df, target_col)
    preprocess_s = time.time() - t0

    model = TabPFNPredictor(
        task_type=task_type,
        device=device,
        model_path=model_path,
        categorical_indices=data.categorical_indices,
        max_train_samples=max_train_samples,
        batch_size=batch_size,
    )

    t1 = time.time()
    model.fit(data.X_train, data.y_train)
    fit_s = time.time() - t1

    t2 = time.time()
    preds = model.predict(data.X_test)
    predict_s = time.time() - t2

    if task_type == REGRESSION:
        metrics = _regression_metrics(data.y_test, preds)
        metrics["baseline_mean_mae"] = float(
            np.mean(np.abs(data.y_test - np.mean(data.y_train)))
        )
    else:
        metrics = _binary_metrics(data.y_test, preds)

    return {
        "predictions": preds,
        "metrics": metrics,
        "timings_s": {"preprocess": preprocess_s, "fit": fit_s, "predict": predict_s},
        "n_train": int(data.X_train.shape[0]),
        "n_test": int(data.X_test.shape[0]),
        "n_features": int(data.X_train.shape[1]),
        "feature_names": data.feature_names,
        "categorical_indices": data.categorical_indices,
        "col_to_stype": data.col_to_stype,
    }
