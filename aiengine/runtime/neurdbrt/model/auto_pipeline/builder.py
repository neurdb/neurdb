import shutil
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from neurdbrt.dataloader import StreamingDataSet
from neurdbrt.log import logger
from neurdbrt.utils.date import time_since

from ..base import BuilderBase
from .run import evaluate_single_dataset, Info
from .config import default_config as conf


class AutoPipelineBuilder(BuilderBase):
    def __init__(self, args):
        super().__init__(args)
        self._logger = logger.bind(model="CtxPipe")

        self._parse_noutput()

    def _parse_noutput(self):
        """for binary classification, should directly use pos/neg to indicate 1/0"""
        if self._args.noutput == 2:
            self._args.noutput = 1

    # def _load_model(self):
    # print(f"[_load_model]: Loading model from {self._args.state_dict_path}")

    async def train(
        self,
        train_loader: StreamingDataSet,
        val_loader: StreamingDataSet,
        test_loader: StreamingDataSet,
        epoch: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int,
        feature_names: List[str],
        target_name: str,
    ):
        logger = self._logger.bind(task="train")

        # create model
        # self._load_model()

        # logger.info("model created with args", **vars(self._args))

        start_time = time.time()

        data = []

        # consume all data so that it does not block the event loop
        for epoch in range(self._args.epoch):
            logger.info("Epoch start", curr_epoch=epoch, end_at_epoch=self._args.epoch)

            batch_idx = -1
            async for batch in train_loader:
                batch_idx += 1

                data.extend(batch)

                if batch_idx + 1 == train_batch_num:
                    break

            batch_idx = -1
            async for _ in val_loader:
                batch_idx += 1

                if batch_idx + 1 == eva_batch_num:
                    break

            batch_idx = -1
            async for _ in test_loader:
                batch_idx += 1

                if batch_idx + 1 == test_batch_num:
                    break

        logger.info("all data collected", len=len(data))
        
        print(data[0])

        # dummy model
        self._model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        self._logger.info("Train end", time=time_since(since=start_time))

        if isinstance(train_loader, StreamingDataSet):
            self._logger.info(
                f"streaming dataloader time usage = {train_loader.total_time_fetching}"
            )

    async def inference(
        self,
        data_loader: StreamingDataSet,
        inf_batch_num: int,
        feature_names: List[str],
        target_name: str,
    ):
        logger = self._logger.bind(task="inference")
        print(f"begin inference for {inf_batch_num} batches ")
        # if this is to load model from the dict,
        # if self._args.state_dict_path:
        #     print("loading model from state dict")
        #     # self._load_model()
        #     # self._model.load_state_dict(torch.load(self._args.state_dict_path))
        #     logger.info("model loaded", state_dict_path=self._args.state_dict_path)
        # else:
        #     print("loading model from database")

        data = []

        start_time = time.time()
        predictions = []
        with torch.no_grad():
            batch_idx = -1
            async for batch in data_loader:
                batch_idx += 1

                data.extend(batch)

                # y = self._model(batch)
                # predictions.append(y.cpu().numpy().tolist())

                # logger.info(f"done batch for {batch_idx}, total {inf_batch_num} ")

                # dummy data
                predictions.append(np.random.rand(len(batch)).tolist())

                if batch_idx + 1 == inf_batch_num:
                    break

        logger.info("all data collected", len=len(data))

        logger.info(f"Done inference for {inf_batch_num} batches ")
        logger.info("---- Inference end ---- ", time=time_since(since=start_time))

        return predictions
