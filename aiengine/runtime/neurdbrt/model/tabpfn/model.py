"""Thin wrapper around TabPFN v2 for in-context tabular prediction.

TabPFN is a prior-fitted transformer that "trains" by conditioning on the
training set in-context (no gradient training) and predicts in a single forward
pass. This wrapper:

- selects ``TabPFNRegressor`` / ``TabPFNClassifier`` by task type,
- subsamples the training context to TabPFN's limit (default 10k rows),
- passes categorical feature indices through,
- predicts in batches (regression -> values, binary -> P(class=1)).

``tabpfn`` and ``torch`` are imported lazily so this module can be imported in
environments without them (the call sites raise a clear error instead).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

REGRESSION = "regression"
BINARY = "binary_classification"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TabPFNPredictor:
    def __init__(
        self,
        task_type: str = REGRESSION,
        *,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        categorical_indices: Optional[Sequence[int]] = None,
        max_train_samples: int = 10_000,
        batch_size: int = 4096,
        random_state: int = 42,
    ):
        if task_type not in (REGRESSION, BINARY):
            raise ValueError(
                f"task_type must be {REGRESSION!r} or {BINARY!r}, got {task_type!r}"
            )
        self.task_type = task_type
        self.device = device or _default_device()
        self.model_path = model_path
        self.categorical_indices = list(categorical_indices) if categorical_indices else []
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build(self):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "tabpfn is not installed in this environment. Install it "
                "(pip install tabpfn) or run under the tabpfn conda env."
            ) from e

        kwargs = {"device": self.device}
        if self.model_path:
            kwargs["model_path"] = self.model_path
        if self.categorical_indices:
            kwargs["categorical_features_indices"] = self.categorical_indices

        cls = TabPFNRegressor if self.task_type == REGRESSION else TabPFNClassifier
        return cls(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNPredictor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]
        self._model = self._build()
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabPFNPredictor.predict called before fit")
        X = np.asarray(X, dtype=float)
        out: List[np.ndarray] = []
        for i in range(0, X.shape[0], self.batch_size):
            xb = X[i : i + self.batch_size]
            if self.task_type == REGRESSION:
                out.append(np.asarray(self._model.predict(xb)))
            else:
                out.append(np.asarray(self._model.predict_proba(xb))[:, 1])
        if not out:
            return np.empty((0,), dtype=float)
        return np.concatenate(out, axis=0)
