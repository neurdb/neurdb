import torch
import torch.nn as nn
import torch.optim as optim
from models.base.builder import BuilderBase
from models.mlp_clf.model.model import MLP
from sklearn.metrics import accuracy_score


class MLPBuilder(BuilderBase):
    pass
