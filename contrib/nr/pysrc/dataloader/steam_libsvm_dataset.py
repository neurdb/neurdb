from cache.data_cache import DataCache, Bufferkey
from typing import Tuple, Dict, List
import time


class StreamingDataSet:
    def __init__(self, data_cache: DataCache, data_key: Bufferkey = Bufferkey.TRAIN_KEY, all_batch_num: int = -1):
        self.data_cache = data_cache
        self.data_key = data_key

        self.all_batches = all_batch_num

    def current_statistics(self) -> Tuple:
        return self.data_cache.dataset_statistics

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, List]:
        """
        Wait until the next data is available.
        :return: current batch data
        """
        waiting_message_printed = False
        while True:
            batch_data = self.data_cache.get(self.data_key)
            if batch_data is not None:
                print(f"Get data from buffer of {self.data_key} !!")
                return batch_data
            if not waiting_message_printed:
                print(f"The buffer of {self.data_key} is not filled yet ! Waiting...")
                waiting_message_printed = True
            time.sleep(0.1)

    def __len__(self):
        return self.all_batches


def libsvm_dataloader(batch_size: int, data_loader_worker: int, data_loader: StreamingDataSet, batch_per_epoch: int):
    data_loader.all_batches = batch_per_epoch
    val_data_loader = StreamingDataSet(data_cache=data_loader.data_cache, data_key=Bufferkey.EVALUATE_KEY,
                                       all_batch_num=batch_per_epoch)
    test_data_loader = StreamingDataSet(data_cache=data_loader.data_cache, data_key=Bufferkey.TEST_KEY,
                                        all_batch_num=batch_per_epoch)
    nfeat, nfields = data_loader.current_statistics()

    return data_loader, val_data_loader, test_data_loader, nfields, nfeat


def build_inference_loader(data_loader_worker: int, data_loader: StreamingDataSet, batch_size: int = -1,
                           all_batch_num: int = -1):
    data_loader.all_batches = all_batch_num
    nfeat, nfields = data_loader.current_statistics()
    return data_loader, nfields, nfeat
