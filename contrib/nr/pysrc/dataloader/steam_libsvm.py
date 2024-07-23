import time
from cache.data_cache import DataCache, Bufferkey
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class StreamingDataSet(Dataset):
    def __init__(self, data_cache: DataCache, data_key: str = Bufferkey.TRAIN_KEY):
        self.data_cache = data_cache
        self.data_key = data_key
        self.last_fetch_time = 0
        self.end_signal = "end_position"

    def __iter__(self):
        return self

    def __next__(self) -> Optional[dict]:
        """
        Fetch data in for loop
        :return: a batch of data in dict
        """
        print("compute time = ", time.time() - self.last_fetch_time)
        self.last_fetch_time = time.time()
        data = self.data_cache.get(self.data_key)
        if data is None or self.end_signal in data:
            print(f"[StreamingDataSet] StopIteration {self.data_key}")
            raise StopIteration
        else:
            print(f"[StreamingDataSet] fetching {self.data_key} data...")
            return data

    def __len__(self) -> int:
        with self.data_cache.lock:
            return len(self.data_cache.cache[self.data_key])


def libsvm_dataloader(batch_size: int, data_loader_worker: int, data_loader: StreamingDataSet):
    train_loader = DataLoader(data_loader,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=data_loader_worker,
                              pin_memory=True)

    val_loader = DataLoader(data_loader,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=data_loader_worker,
                            pin_memory=True)

    test_loader = DataLoader(data_loader,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=data_loader_worker,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, nfields, nfeat


def build_inference_loader(data_loader_worker: int, data_loader: StreamingDataSet, batch_size=0):
    loader = DataLoader(data_loader,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=data_loader_worker,
                        pin_memory=True)
    return loader, nfields, nfeat
