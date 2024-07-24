import time
from cache.data_cache import DataCache, Bufferkey
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class StreamingDataSet(Dataset):
    def __init__(self, data_cache: DataCache, data_key: Bufferkey = Bufferkey.TRAIN_KEY):
        self.data_cache = data_cache
        self.data_key = data_key

    def current_statistics(self) -> Tuple:
        """
        :return:  _nfeat, _nfield
        """
        return self.data_cache.dataset_statistics

    def __iter__(self):
        return self

    def __next__(self) -> Optional[dict]:
        """
        Fetch data in for loop
        :return: a batch of data in dict
        """
        data = self.data_cache.get(self.data_key)
        print(f"[StreamingDataSet] fetching {self.data_key} data...")
        return data

    def __len__(self) -> int:
        with self.data_cache.lock:
            return len(self.data_cache.cache[self.data_key])


def libsvm_dataloader(batch_size: int, data_loader_worker: int, data_loader: StreamingDataSet):
    # share the data cache
    train_loader = DataLoader(data_loader,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=data_loader_worker,
                              pin_memory=True)

    val_data_loader = StreamingDataSet(data_cache=data_loader.data_cache, data_key=Bufferkey.EVALUATE_KEY)
    test_data_loader = StreamingDataSet(data_cache=data_loader.data_cache, data_key=Bufferkey.TEST_KEY)

    val_loader = DataLoader(val_data_loader,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=data_loader_worker,
                            pin_memory=True)

    test_loader = DataLoader(test_data_loader,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=data_loader_worker,
                             pin_memory=True)

    nfeat, nfields = data_loader.current_statistics()

    return train_loader, val_loader, test_loader, nfields, nfeat


def build_inference_loader(data_loader_worker: int, data_loader: StreamingDataSet, batch_size=0):
    loader = DataLoader(data_loader,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=data_loader_worker,
                        pin_memory=True)
    nfeat, nfields = data_loader.current_statistics()
    return loader, nfields, nfeat
