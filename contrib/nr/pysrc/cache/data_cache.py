import threading
from typing import Optional
from enum import Enum


# Define global variables for cache keys
class Bufferkey(Enum):
    TRAIN_KEY = "train"
    EVALUATE_KEY = "evaluate"
    TEST_KEY = "test"
    INFERENCE_KEY = "inference"

    @staticmethod
    def get_key_by_value(value):
        for key in Bufferkey:
            if key.value == value:
                return key
        return None


class DataCache:
    def __init__(self, dataset_name: str, maxsize=20):
        self.current_dataset_name = dataset_name
        self.lock = threading.Lock()
        self.cache = {
            Bufferkey.TRAIN_KEY: [],
            Bufferkey.EVALUATE_KEY: [],
            Bufferkey.TEST_KEY: [],
            Bufferkey.INFERENCE_KEY: []
        }
        self.maxsize = maxsize

        # for dataset_statistics of current datasets
        self._nfeat = None
        self._nfield = None

    @property
    def dataset_statistics(self):
        return self._nfeat, self._nfield

    @dataset_statistics.setter
    def dataset_statistics(self, statistics):
        nfeat, nfield = statistics
        self._nfeat = nfeat
        self._nfield = nfield

    def set(self, key: Bufferkey, value: dict) -> bool:
        with self.lock:
            if key in self.cache and len(self.cache[key]) < self.maxsize:
                self.cache[key].append(value)
                return True
            return False

    def get(self, key: Bufferkey) -> Optional[dict]:
        with self.lock:
            if key in self.cache and len(self.cache[key]) > 0:
                return self.cache[key].pop(0)
            return None

    def is_full(self) -> Optional[Bufferkey]:
        """
        Check if the queue is full
        :return: Bufferkey.TRAIN_KEY etc
        """
        with self.lock:
            for key_value, values in self.cache.items():
                if len(values) < self.maxsize:
                    return key_value
            return None

    def is_empty(self):
        with self.lock:
            return all(len(values) == 0 for values in self.cache.values())
