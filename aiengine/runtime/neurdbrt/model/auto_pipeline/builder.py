import pickle
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
from auto_pipeline.config import default_config as conf
from auto_pipeline.ctxpipe.env.primitives.primitive import Primitive
from auto_pipeline.ctxpipe.env.primitives.predictor import LogisticRegressionPrim


class AutoPipelineBuilder(BuilderBase):
    def __init__(self, args):
        super().__init__(args)
        self._logger = logger.bind(model="CtxPipe")

        self._parse_noutput()

    def _parse_noutput(self):
        """for binary classification, should directly use pos/neg to indicate 1/0"""
        if self._args.noutput == 2:
            self._args.noutput = 1

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

        start_time = time.time()

        data = []

        # consume all data so that it does not block the event loop
        for epoch in range(self._args.epoch):
            logger.info("Epoch start", curr_epoch=epoch, end_at_epoch=self._args.epoch)

            batch_idx = -1
            async for batch in train_loader:
                batch_idx += 1

                features = batch["value"]
                label = batch["y"]
                for i in range(len(features)):
                    data.append([*features[i], label[i]])

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

        logger.debug("all data collected", len=len(data), first=data[0])

        df = pd.DataFrame(data, columns=[*feature_names, target_name])
        logger.info("data frame created", len=len(df), df=df)

        conf.data_in_memory = True
        conf.data = df
        conf.data_info = {
            "label_name": target_name,
            "label_index": conf.data.columns.get_loc(target_name),
        }
        pipeline = evaluate_single_dataset(
            Info(
                aipipe_core_prefix=f"{conf.exp_dir}/.tmp",
                result_prefix=f"{conf.exp_dir}/.tmp/result",
                dataset_prefix="",
            ),
            start=32000,
            end=32000,
            dry_run=True,
        )
        if pipeline is not None:
            logger.info("pipeline", pipeline=pipeline)
            self._model = [pickle.dumps(s) for s in pipeline]
        else:
            raise ValueError("pipeline is None")

        shutil.rmtree(f"{conf.exp_dir}/.tmp")

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
        logger.debug(f"begin inference for {inf_batch_num} batches")

        data = []

        start_time = time.time()
        predictions = []
        with torch.no_grad():
            batch_idx = -1
            async for batch in data_loader:
                batch_idx += 1

                features = batch["value"]
                for i in range(len(features)):
                    data.append(features[i])

                ### dummy data
                # predictions.append(np.random.rand(len(batch)).tolist())

                if batch_idx + 1 == inf_batch_num:
                    break

        logger.debug("all data collected", len=len(data))

        sequence: List[Primitive] = [pickle.loads(s) for s in self._model]

        logger.info(f"sequence: {sequence}")

        df = pd.DataFrame(data, columns=feature_names).infer_objects()
        logger.info("data frame created", len=len(df), df=df)

        for i in range(len(sequence) - 1):
            df = sequence[i].transform_x(df)

        predictor: LogisticRegressionPrim = sequence[-1]
        prediction = predictor.predict(df)[:, 1] - 0.5
        print(prediction)
        
        predictions = [prediction]

        logger.info(f"Done inference for {inf_batch_num} batches ")
        logger.info("---- Inference end ---- ", time=time_since(since=start_time))

        return predictions
