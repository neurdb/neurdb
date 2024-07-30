import threading
import time
from cache import DataCache
import torch
from typing import Callable


class LibSvmDataDispatcher:
    """
    LibSvmDataDispatcher monitor the cahce and ask data from UDF to fill the cache.
    """

    def __init__(self, data_cache: DataCache = None):
        """
        Initialize the LibSvmDataDispatcher with an optional data cache.
        :param data_cache: The data cache it is currently handling.
        """
        self.data_cache = data_cache
        self.client_id = None

        self.thread = None
        self.stop_event = threading.Event()

    def bound_client_to_cache(self, data_cache: DataCache, client_id: str):
        """
        Bind a client to the data cache.
        :param data_cache: The data cache to bind to.
        :param client_id: The client ID.
        """
        self.data_cache = data_cache
        self.client_id = client_id

    # ------------------------- data prepceossing -------------------------

    def batch_preprocess(self, data: str):
        """
        Preprocess a batch of data from LibSVM format.
        :param data: The data in LibSVM format.
        :return: A dictionary with processed 'id', 'value', and 'y' tensors.
        """
        print(f"[Data Preprocessing]: Preprocessing started...")
        max_nfileds = self.data_cache.dataset_statistics[1]
        print(f"[Data Preprocessing]: max_nfileds = {max_nfileds}")
        data = data.split("\n")

        sample_lines = 0
        ids_list = []
        values_list = []
        labels_list = []
        print(f"[Data Preprocessing]: # {len(data)} data samples to be loaded...")
        for line in data:
            if not line:
                continue  # skip empty lines
            columns = line.strip().split(" ")
            pairs = [list(map(int, pair.split(":"))) for pair in columns[1:]]
            ids, values = zip(*pairs) if pairs else ([], [])
            ids_list.append(ids)
            values_list.append(values)
            labels_list.append(float(columns[0]))
            sample_lines += 1
        print(f"[Data Preprocessing]: # {sample_lines} data samples loaded...")
        nsamples = sample_lines
        feat_id = torch.zeros((nsamples, max_nfileds), dtype=torch.long)
        feat_value = torch.zeros((nsamples, max_nfileds), dtype=torch.float)
        y = torch.tensor(labels_list, dtype=torch.float)
        print(f"[Data Preprocessing]: Creating tensors...")

        for i in range(nsamples):
            try:
                ids = ids_list[i]
                values = values_list[i]
                feat_id[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                feat_value[i, :len(values)] = torch.tensor(values, dtype=torch.float)
            except Exception as e:
                print(f"[Data Preprocessing]: Incorrect data format in sample {i}! Error: {e}")
        print(f"[Data Preprocessing]: # {nsamples} data samples loaded successfully.")

        return {"id": feat_id, "value": feat_value, "y": y}

    def add(self, data: str):
        """
        Add data to the cache under the given key.
        :param data: The dataset in LibSVM format.
        :return: True if the data was added successfully, False otherwise.
        """
        print(f"[LibSvmDataDispatcher] add data to cache...")
        batch_data = self.batch_preprocess(data)
        if self.data_cache.add(batch_data):
            return True
        else:
            return False

    # ------------------------- threading -------------------------

    def start(self, emit_request_data: Callable[[str], None]):
        """
        Start the background thread to manage data dispatch.
        :param emit_request_data: The function to call for requesting data.
        """
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._background_thread, args=(emit_request_data,)
        )
        self.thread.daemon = True
        self.thread.start()

    def _background_thread(self, emit_request_data):
        """
        The background thread function for managing data dispatch.
        :param emit_request_data: The function to call for requesting data.
        """
        print("[LibSvmDataDispatcher] thread started...")
        while not self.stop_event.is_set():
            # stop if meets
            isStop = self.data_cache.time_to_stop()
            if isStop:
                print(
                    f"[LibSvmDataDispatcher] data_cache {self.data_cache.current_batch_num} "
                    f"meets {self.data_cache.total_batch_num}, stop asking data."
                )
                break
            # consume if not full
            isfull = self.data_cache.is_full()
            if not isfull:
                print(f"queue current length {self.data_cache.current_len()}, emit req to client_id {self.client_id} ")
                emit_request_data(self.client_id)
            time.sleep(0.2)

    def stop(self):
        """
        Stop the background thread.
        """
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
