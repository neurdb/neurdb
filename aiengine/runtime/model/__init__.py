from .base import (
    CategoricalFeature,
    Model,
    ModelKind,
    ModelSpec,
    TaskType,
    TrainProtocol,
)
from .dispatcher import ModelDispatcher
from .tokenizer import FeatureTokenizer

# Importing the family packages registers their boxes with the dispatcher
# (each family self-guards its optional dependencies).
from . import sklearn_family  # noqa: F401  isort: skip
from . import torch_family  # noqa: F401  isort: skip
from .sklearn_family import MatrixInput, SklearnMiddleware
from .torch_family import ModelInput, TorchMiddleware
from .trainer import PipelineTrainer

__all__ = [
    "CategoricalFeature",
    "Model",
    "ModelKind",
    "ModelSpec",
    "TaskType",
    "TrainProtocol",
    "ModelDispatcher",
    "FeatureTokenizer",
    "MatrixInput",
    "SklearnMiddleware",
    "ModelInput",
    "TorchMiddleware",
    "PipelineTrainer",
]
