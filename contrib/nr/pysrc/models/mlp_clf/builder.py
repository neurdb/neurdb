import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from connection.pg_connect import LoadedDataset
from models.mlp_clf.model.model import MLP
from models.base.builder import BuilderBase


class MLPBuilder(BuilderBase):
    pass
