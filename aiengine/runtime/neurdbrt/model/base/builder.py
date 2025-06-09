from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
from neurdbrt.dataloader.stream_libsvm_dataset import StreamingDataSet
from torch import nn
from torch.utils.data import DataLoader


class BuilderBase(ABC):
    """
    Abstract base class for model builders.
    Any subclass must implement the train_model and evaluate_model methods.

    Args:
        args (argparse.Namespace): The arguments for the model. Will be set to `self._args`.
    """

    def __init__(self, args):
        self._model: nn.Module = None
        self._args = args

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @abstractmethod
    async def train(
        self,
        train_loader: Union[DataLoader, StreamingDataSet],
        val_loader: Union[DataLoader, StreamingDataSet],
        test_loader: Union[DataLoader, StreamingDataSet],
        epoch: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int,
    ):
        """
        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param epoch: num of epoch
        :param train_batch_num: batch in each epoch
        :param eva_batch_num: overall batch
        :param test_batch_num: overall batch
        :return:
        """
        pass

    @abstractmethod
    async def inference(
        self, test_loader: Union[DataLoader, StreamingDataSet], inf_batch_num: int
    ) -> List[List[Any]]:
        """

        :param test_loader:
        :param inf_batch_num:
        :return:
        """
        pass
