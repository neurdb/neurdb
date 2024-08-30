from cache import DataCache, Bufferkey
from typing import Tuple, Dict
from log import logger
import time


class StreamingDataSet:
    def __init__(
        self,
        data_cache: DataCache,
        stage_counts: Dict[Bufferkey, int] = None,
        data_key: Bufferkey = Bufferkey.TRAIN_KEY,
    ):

        self.data_cache = data_cache

        if data_key != Bufferkey.INFERENCE_KEY:
            self.ml_stages = [
                Bufferkey.TRAIN_KEY,
                Bufferkey.EVALUATE_KEY,
                Bufferkey.TEST_KEY,
            ]
            self.stage_counts = stage_counts
        else:
            self.ml_stages = [Bufferkey.INFERENCE_KEY]
            self.stage_counts = stage_counts

        self.current_stage_index = 0
        self.current_stage = self.ml_stages[0]
        self.current_stage_batch_count = 0

        self.total_time_fetching = 0

    def current_statistics(self) -> Tuple:
        return self.data_cache.dataset_statistics

    def __aiter__(self):
        return self

    async def __anext__(self):
        """
        Wait until the next data is available.
        :return: current batch data
        """
        logger.debug(f"[StreamingDataSet]: reading one data from queue...")
        begin_time = time.time()
        batch_data = await self.data_cache.get()
        end_time = time.time()
        self.total_time_fetching += end_time - begin_time
        if batch_data is None:
            # raise to http response
            raise "No data to read after waiting for 10 mins"

        # increase the current stage count
        self.current_stage_batch_count += 1
        if self.current_stage_batch_count >= self.stage_counts[self.current_stage]:
            _pre_stage_for_log = self.current_stage
            # switch to next stage
            self.current_stage_index = (self.current_stage_index + 1) % len(
                self.ml_stages
            )
            self.current_stage = self.ml_stages[self.current_stage_index]
            logger.debug(
                f"[Streaming Dataloader]: stage {_pre_stage_for_log} finished batch "
                f"{self.current_stage_batch_count} and switch to stage {self.current_stage}!!"
            )
            self.current_stage_batch_count = 0

        return batch_data

    def __len__(self):
        # max number of batches in current stage.
        return self.stage_counts[self.current_stage]

    def setup_for_train_task(
        self, train_batch_num: int, eva_batch_num: int, test_batch_num: int
    ):

        self.stage_counts = {
            Bufferkey.TRAIN_KEY: train_batch_num,
            Bufferkey.EVALUATE_KEY: eva_batch_num,
            Bufferkey.TEST_KEY: test_batch_num,
        }

        nfeat, nfields = self.current_statistics()

        return nfields, nfeat

    def setup_for_inference_task(self, inf_batch_num: int = -1):

        self.stage_counts = {
            Bufferkey.INFERENCE_KEY: inf_batch_num,
        }
