"""Concrete sklearn boxes."""

from __future__ import annotations

from typing import Any, Tuple

from ..base import TaskType
from ..dispatcher import ModelDispatcher
from .base import SklearnModel
from .middleware import MatrixColumn


@ModelDispatcher.register("gbdt")
class GradientBoostingModel(SklearnModel):
    """Gradient-boosted trees — the tree-based baseline.

    ``params`` pass through to sklearn (n_estimators, max_depth,
    learning_rate, ...). Integer categorical codes are consumed directly
    as split features, so the column descriptors are not needed.
    """

    def build_estimator(self, columns: Tuple[MatrixColumn, ...]) -> Any:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        if self.spec.task_type is TaskType.REGRESSION:
            return GradientBoostingRegressor(**self.params)
        return GradientBoostingClassifier(**self.params)


@ModelDispatcher.register("logreg")
class LogisticRegressionModel(SklearnModel):
    """Logistic regression — the linear classification baseline.

    Classification only (a regression task fails at build). Raw integer
    categorical codes are NOT ordinal, so the box wraps the estimator in a
    sklearn Pipeline that one-hot-encodes the categorical positions taken
    from the matrix descriptors (``handle_unknown="ignore"`` covers the
    reserved unknown code at inference). ``params`` pass through to
    ``LogisticRegression`` (C, penalty, ...; ``max_iter`` defaults 1000).
    """

    def build_estimator(self, columns: Tuple[MatrixColumn, ...]) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        if self.spec.task_type is TaskType.REGRESSION:
            raise ValueError(
                "model 'logreg' is a classifier; the task's target is "
                "numerical (regression) — use a regression-capable model"
            )

        from data.base import ColumnStype

        params = {"max_iter": 1000, **self.params}
        categorical_idx = [
            j for j, c in enumerate(columns)
            if c.stype is ColumnStype.CATEGORICAL
        ]
        encode = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_idx)],
            remainder="passthrough",
        )
        return Pipeline(
            [("encode", encode), ("classify", LogisticRegression(**params))]
        )
