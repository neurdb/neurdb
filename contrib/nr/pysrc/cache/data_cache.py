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
        """
        Get the Bufferkey enum member corresponding to the given value.
        :param value: The value to search for.
        :return: The corresponding Bufferkey enum member, or None if not found.
        """
        for key in Bufferkey:
            if key.value == value:
                return key
        return None


class DataCache:
    """
    DataCache manages caching of different datasets with thread-safe operations.
    """

    def __init__(self, dataset_name: str, maxsize=20):
        """
        Initialize the DataCache with a dataset name and a maximum cache size.
        :param dataset_name: The name of the dataset.
        :param maxsize: The maximum number of items allowed in the cache for each Bufferkey.
        """
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
        """
        Get the dataset statistics (number of features and fields).
        :return: A tuple (nfeat, nfield) representing the dataset statistics.
        """
        return self._nfeat, self._nfield

    @dataset_statistics.setter
    def dataset_statistics(self, statistics):
        """
        Set the dataset statistics (number of features and fields).
        :param statistics: A tuple (nfeat, nfield) representing the dataset statistics.
        """
        nfeat, nfield = statistics
        self._nfeat = nfeat
        self._nfield = nfield

    def set(self, key: Bufferkey, value: dict) -> bool:
        """
        Add a value to the cache for the given key if the cache is not full.
        :param key: The Bufferkey enum member indicating which cache to add to.
        :param value: The value to add to the cache.
        :return: True if the value was added, False if the cache is full.
        """
        with self.lock:
            if key in self.cache and len(self.cache[key]) < self.maxsize:
                self.cache[key].append(value)
                return True
            return False

    def get(self, key: Bufferkey) -> Optional[dict]:
        """
        Retrieve and remove the oldest value from the cache for the given key.
        :param key: The Bufferkey enum member indicating which cache to retrieve from.
        :return: The oldest value from the cache, or None if the cache is empty.
        """
        with self.lock:
            if key in self.cache and len(self.cache[key]) > 0:
                return self.cache[key].pop(0)
            return None

    def is_full(self) -> Optional[Bufferkey]:
        """
        Check if any cache is full.
        :return: The Bufferkey enum member for the cache that is not full, or None if all caches are full.
        """
        with self.lock:
            for key_value, values in self.cache.items():
                if len(values) < self.maxsize:
                    return key_value
            return None

    def is_empty(self) -> bool:
        """
        Check if all caches are empty.
        :return: True if all caches are empty, False otherwise.
        """
        with self.lock:
            return all(len(values) == 0 for values in self.cache.values())
