from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch.nn as nn


class BuilderBase(ABC):
    """
    Abstract base class for model builders.
    Any subclass must implement the train_model and evaluate_model methods.
    """

    def __init__(self):
        self.model = None
        self._nfeat = None
        self._nfield = None

    @property
    def model_dimension(self):
        """
        Get the model dimensions.
        @return: A tuple containing (nfeat, nfield)
        """
        return self._nfeat, self._nfield

    @model_dimension.setter
    def model_dimension(self, dimensions):
        """
        Set the model dimensions.
        @param dimensions: A tuple containing (nfeat, nfield)
        @return: None
        """
        nfeat, nfield = dimensions
        self._nfeat = nfeat
        self._nfield = nfield

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def inference(self, test_loader: DataLoader):
        """
        Evaluate the model.
        """
        pass
