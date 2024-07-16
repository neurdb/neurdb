import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from connection.pg_connect import LoadedDataset
from apps.mlp_clf.model.model import MLP
from apps.base.builder import BuilderBase


class MLPBuilder(BuilderBase):
    pass

