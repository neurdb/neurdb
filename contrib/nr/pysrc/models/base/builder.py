from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from torch.utils.data import DataLoader
from dataloader.steam_libsvm_dataset import StreamingDataSet


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

    def train(self,
              train_loader: Union[DataLoader, StreamingDataSet],
              val_loader: Union[DataLoader, StreamingDataSet],
              test_loader: Union[DataLoader, StreamingDataSet],
              epochs: int,
              batch_per_epoch: int
              ):
        """

        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param epochs: num of epoch
        :param batch_per_epoch: batch in each epoch
        :return:
        """
        pass

    @abstractmethod
    def inference(self, test_loader: Union[DataLoader, StreamingDataSet], all_batch_num: int) -> List[np.ndarray]:
        """

        :param test_loader:
        :param all_batch_num:
        :return:
        """
        pass
