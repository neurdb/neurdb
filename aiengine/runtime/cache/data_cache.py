import asyncio
from typing import Optional, Tuple
from enum import Enum
from queue import Full, Empty
from logger.logger import logger


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
    todo: manage timeout for even driven task,
    """

    def __init__(self, dataset_name: str, total_batch_num: int, maxsize=80):
        """
        Initialize the DataCache with a dataset name and a maximum cache size.
        :param dataset_name: The name of the dataset.
        :param maxsize: The maximum number of items allowed in the cache.
        """
        self.current_dataset_name = dataset_name
        # self.lock = threading.Lock()
        self.queue = asyncio.Queue(maxsize=maxsize)

        self.maxsize = maxsize
        self.total_batch_num = total_batch_num
        self.current_batch_num = 0

        # for dataset_statistics of current datasets
        self._nfeat = None
        self._nfield = None

    @property
    def dataset_statistics(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the dataset statistics (number of features and fields).
        :return: A tuple (nfeat, nfield) representing the dataset statistics.
        """
        return self._nfeat, self._nfield

    @dataset_statistics.setter
    def dataset_statistics(self, statistics: Tuple[int, int]):
        """
        Set the dataset statistics (number of features and fields).
        :param statistics: A tuple (nfeat, nfield) representing the dataset statistics.
        """
        nfeat, nfield = statistics
        self._nfeat = nfeat
        self._nfield = nfield

    # ------------------------- data operation -------------------------

    async def add(self, value) -> bool:
        """
        Add a value to the right of the queue if the queue is not full.
        :param value: The value to add to the queue.
        :return: True if the value was added, False if the queue is full.
        """
        # with self.lock:
        try:
            await self.queue.put(value)
            self.current_batch_num += 1
            return True
        except Full:
            logger.debug(
                "Queue is full, and item could not be added within the timeout period."
            )
            return False

    async def get(self) -> Optional[dict]:
        """
        Retrieve and remove the oldest value from the queue (read from left).
        :return: The oldest value from the queue, or None if the queue is empty.
        """
        try:
            logger.debug("getting from data cache", remaining=self.queue.qsize())
            value = await self.queue.get()  # Block up to 5 seconds
            return value
        except Empty:
            logger.debug(
                "Queue was empty and no item was available within the timeout period."
            )
            return None

    def is_full(self) -> bool:
        """
        Check if the queue is full.
        :return: True if the queue is full, False otherwise.
        """
        return self.queue.full()

    def time_to_stop(self) -> bool:
        """
        Check if the queue is full.
        :return: True if the total batch meet
        """
        return self.current_batch_num == self.total_batch_num

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        :return: True if the queue is empty, False otherwise.
        """
        return self.queue.empty()

    def current_len(self):
        return self.queue.qsize()
