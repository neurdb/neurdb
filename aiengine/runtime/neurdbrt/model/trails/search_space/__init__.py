"""
Search space models for zero-cost NAS

Provides custom MLP and ResNet models with flexible architectures.
"""

from .mlp import TrailsMLP
from .resnet import TrailsResNet
from .space_base import BaseSearchSpace

__all__ = ["TrailsMLP", "TrailsResNet", "BaseSearchSpace"]
