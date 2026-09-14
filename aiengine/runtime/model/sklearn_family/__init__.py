"""Sklearn model family: FIT_ONCE boxes over design matrices.

Importing this package registers its boxes with ModelDispatcher.
``SklearnMiddleware`` packages the pipeline's ``EncodedBatch`` stream into
the family's ``MatrixInput`` (self-describing: schema-declared column
types travel with the matrix).
"""

from .base import SklearnModel
from .middleware import MatrixColumn, MatrixInput, MatrixStep, SklearnMiddleware
from .models import GradientBoostingModel, LogisticRegressionModel

__all__ = [
    "SklearnModel",
    "SklearnMiddleware",
    "MatrixColumn",
    "MatrixInput",
    "MatrixStep",
    "GradientBoostingModel",
    "LogisticRegressionModel",
]
