import asyncio
import json

import threading
from websocket_sender import WebsocketSender

from cache import DataCache

from dataloader.preprocessing import (
    libsvm_batch_preprocess_id_only,
)
from log import logger
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

        self.task = None
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

    def bound_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Bind the data dispatcher to an event loop.
        :param loop: The event loop to bind to.
        """
        self.loop = loop

    # ------------------------- threading -------------------------
    def start(self):
        """
        Start the background thread to manage data dispatch.
        :param emit_request_data: The function to call for requesting data.
        """
        logger.debug("starting dispatcher")
        # Ensure event is clear before starting the thread, not wating somewhere
        self.stop_event.clear()
        self.full_event.clear()

    def stop(self):
        """
        Stop the background thread.
        """
        logger.debug("stopping dispatcher")
        # if self.thread is not None:
        self.stop_event.set()  # Signal the thread to stop
        self.full_event.set()  # Wake up the thread if it's waiting
        # try:
        #     self.thread.join(
        #         timeout=10
        #     )  # Wait up to 10 seconds for the thread to stop
        # except Exception as e:
        #     logger.debug(
        #         f"[LibSvmDataDispatcher] Exception while stopping thread: {e}"
        #     )
        # finally:
        #     self.thread = None

    def background_task(self):
        """
        The background task function for managing data dispatch.
        """
        logger.debug("[LibSvmDataDispatcher] ----- Background task started  ----- ")

        # Send the initial request to trigger the process
        logger.debug(
            f"[LibSvmDataDispatcher] Queue current length {self.data_cache.current_len()}, "
            f"emitting initial request to client_id {self.client_id}"
        )

        asyncio.run_coroutine_threadsafe(
            websocket_emit_request_data(self.client_id),
            self.loop,
        )

        while not self.stop_event.is_set():
            # Check if we need to stop
            if self.data_cache.time_to_stop():
                logger.debug(
                    f"[LibSvmDataDispatcher] Data cache meets total batch num, stopping."
                )
                break

            # Wait up to 10 minutes
            if self.full_event.wait(timeout=600):
                # Emit request data
                logger.debug(
                    f"[LibSvmDataDispatcher] Sender wake up, queue current length {self.data_cache.current_len()}, "
                    f"emitting request to client_id {self.client_id}"
                )

                asyncio.run_coroutine_threadsafe(
                    websocket_emit_request_data(self.client_id),
                    self.loop,
                )
                # Reset the event
                self.full_event.clear()
            else:
                logger.debug(
                    f"[LibSvmDataDispatcher] No data available after waiting 10 mins."
                )
                self.full_event.clear()
                self.stop()

    # ------------------------- data operation -------------------------

    async def add(self, data: str):
        """
        Add data to the cache.
        :param data: The dataset in LibSVM format.
        """
        logger.debug(f"[LibSvmDataDispatcher] receive data, adding data to cache...")

        # Record the start time
        start_time = time.time()

        # Process the data
        _nfields = self.data_cache.dataset_statistics[1]
        batch_data = libsvm_batch_preprocess_id_only(data, _nfields)

        # Record the end time
        end_time = time.time()
        preprocessing_time = end_time - start_time

        # Update the aggregation variables
        self.total_preprocessing_time += preprocessing_time

        # Add the processed data to the cache
        if await self.data_cache.add(batch_data):
            # Notify that new data is available
            self.full_event.set()
            logger.debug(
                f"[LibSvmDataDispatcher]: Data added, cur length = {self.data_cache.current_len()}"
            )
        else:
            logger.debug(
                f"[LibSvmDataDispatcher]: stopping dispacher threads, no data to add after waiting for 10 mins"
            )
            # self.stop()


async def websocket_emit_request_data(session_id: str):
    await WebsocketSender.send(
        json.dumps(
            {
                "version": 1,
                "event": "request_data",
                "sessionId": session_id,
                "startBatchId": 0,
                "nBatch": 1,
            }
        )
    )
