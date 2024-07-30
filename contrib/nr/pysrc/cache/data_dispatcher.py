import threading
from cache import DataCache
from typing import Callable
from dataloader.preprocessing import libsvm_batch_preprocess
from logger.logger import logger
import time


class LibSvmDataDispatcher:
    """
    LibSvmDataDispatcher monitors the cache and requests data to fill the cache.
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

        # Initialize aggregation variables
        self.total_preprocessing_time = 0.0

    def bound_client_to_cache(self, data_cache: DataCache, client_id: str):
        """
        Bind a client to the data cache.
        :param data_cache: The data cache to bind to.
        :param client_id: The client ID.
        """
        self.data_cache = data_cache
        self.client_id = client_id

    # ------------------------- threading -------------------------

    def start(self, emit_request_data: Callable[[str], None]):
        """
        Start the background thread to manage data dispatch.
        :param emit_request_data: The function to call for requesting data.
        """
        # Ensure event is clear before starting the thread, not wating somewhere
        self.stop_event.clear()
        self.full_event.clear()

        self.thread = threading.Thread(
            target=self._background_thread, args=(emit_request_data,)
        )
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """
        Stop the background thread.
        """
        if self.thread is not None:
            self.stop_event.set()  # Signal the thread to stop
            self.full_event.set()  # Wake up the thread if it's waiting
            try:
                self.thread.join(timeout=10)  # Wait up to 10 seconds for the thread to stop
            except Exception as e:
                logger.debug(f"[LibSvmDataDispatcher] Exception while stopping thread: {e}")
            finally:
                self.thread = None

    def _background_thread(self, emit_request_data: Callable[[str], None]):
        """
        The background thread function for managing data dispatch.
        :param emit_request_data: The function to call for requesting data.
        """
        logger.debug("[LibSvmDataDispatcher] ----- Background thread started  ----- ")

        # Send the initial request to trigger the process
        logger.debug(
            f"[LibSvmDataDispatcher] Queue current length {self.data_cache.current_len()}, "
            f"emitting initial request to client_id {self.client_id}")
        emit_request_data(self.client_id)

        while not self.stop_event.is_set():
            # Check if we need to stop
            if self.data_cache.time_to_stop():
                logger.debug(f"[LibSvmDataDispatcher] Data cache meets total batch num, stopping.")
                break

            # Wait up to 10 minutes
            if self.full_event.wait(timeout=600):
                # Emit request data
                logger.debug(
                    f"[LibSvmDataDispatcher] Sender wake up, queue current length {self.data_cache.current_len()}, "
                    f"emitting request to client_id {self.client_id}")
                emit_request_data(self.client_id)
                # Reset the event
                self.full_event.clear()
            else:
                logger.debug(f"[LibSvmDataDispatcher] No data available after waiting 10 mins.")
                self.stop()

        # Ensure that the event is cleared if the thread stops
        self.full_event.clear()

    # ------------------------- data operation -------------------------

    def add(self, data: str):
        """
        Add data to the cache.
        :param data: The dataset in LibSVM format.
        """
        logger.debug(f"[LibSvmDataDispatcher] receive data, adding data to cache...")

        # Record the start time
        start_time = time.time()

        # Process the data
        _nfields = self.data_cache.dataset_statistics[1]
        batch_data = libsvm_batch_preprocess(data, _nfields)

        # Record the end time
        end_time = time.time()
        preprocessing_time = end_time - start_time

        # Update the aggregation variables
        self.total_preprocessing_time += preprocessing_time

        # Add the processed data to the cache
        if self.data_cache.add(batch_data):
            logger.debug(f"[LibSvmDataDispatcher]: added data done, cur length = {self.data_cache.current_len()}")
        else:
            self.stop()

        # Notify that new data is available
        self.full_event.set()
