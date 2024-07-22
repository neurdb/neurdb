import time
from cache.data_cache import DataCache
from typing import Optional


class StreamingDataLoader:
    def __init__(self, data_cache: DataCache, data_key: str):
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
            print(f"[StreamingDataLoader] StopIteration {self.data_key}")
            raise StopIteration
        else:
            print(f"[StreamingDataLoader] fetching {self.data_key} data...")
            return data

    def __len__(self) -> int:
        with self.data_cache.lock:
            return len(self.data_cache.cache[self.data_key])
