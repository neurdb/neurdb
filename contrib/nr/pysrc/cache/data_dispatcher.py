import threading
from cache import DataCache
from typing import Callable
from dataloader.preprocessing import libsvm_batch_preprocess


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
        self.full_event = threading.Event()

    def bound_client_to_cache(self, data_cache: DataCache, client_id: str):
        """
        Bind a client to the data cache.
        :param data_cache: The data cache to bind to.
        :param client_id: The client ID.
        """
        self.data_cache = data_cache
        self.client_id = client_id

    # ------------------------- data operation -------------------------

    def add(self, data: str):
        """
        Add data to the cache under the given key.
        :param data: The dataset in LibSVM format.
        :return: True if the data was added successfully, False otherwise.
        """
        print(f"[LibSvmDataDispatcher] add data to cache...")
        # todo: make the batch_processing method configurable
        _nfiled = self.data_cache.dataset_statistics[1]
        batch_data = libsvm_batch_preprocess(data, _nfiled)
        self.data_cache.add(batch_data)
        self.full_event.set()

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
            # Check if we need to stop
            if self.data_cache.time_to_stop():
                print(f"[LibSvmDataDispatcher] data_cache meets total batch num, stopping.")
                break

            # Wait until the event is set
            self.full_event.wait()

            # Emit request data
            print(
                f"Queue current length {self.data_cache.current_len()}, emitting request to client_id {self.client_id}")
            emit_request_data(self.client_id)

            # Reset the event
            self.full_event.clear()

    def stop(self):
        """
        Stop the background thread.
        """
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
