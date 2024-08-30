import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..base.builder import BuilderBase
from .model.model import MLP


class MLPBuilder(BuilderBase):
    pass
