import ast
import asyncio
import functools
import concurrent.futures
import json
from threading import Thread
from quart import Quart
from quart.utils import run_sync
import socketio
from cache import LibSvmDataDispatcher, DataCache, ContextStates
from logger.logger import logger


def create_sio_app(quart_app: Quart) -> socketio.AsyncServer:
    sio = socketio.AsyncServer(
        async_mode="asgi",
        ping_timeout=30,
        ping_interval=5,
        # logger=True,
        # engineio_logger=True,
    )

    @sio.event
    async def connect(sid, environ):
        """
        Handle client connection event.
        Store the client session ID and notify the client.
        """
        logger.debug("event: connect", sid=sid)

        quart_app.config["clients"][sid] = sid

        logger.info(f"Client connected: {sid}")
        _current_clients = quart_app.config["clients"]
        logger.debug(f"Current registered clients: {_current_clients}")
        await sio.emit("connection", {"sid": sid}, room=sid)

    @sio.event
    async def disconnect(sid):
        """
        Handle client disconnection event.
        Remove the client session ID and associated data from the server.
        """
        logger.debug("event: disconnect", sid=sid)

        try:
            logger.info(
                f"[socket: Discinnect & Recording] : {sid} Client disconnected: "
            )
            quart_app.config["clients"].pop(sid, None)
            quart_app.config["data_cache"].remove(sid)

            for dataset_name, ele in quart_app.config["dispatchers"].get(sid).items():
                print(
                    f"[socket: Discinnect & Recording] dataset {dataset_name}, sid {sid} time usage {ele.total_preprocessing_time}"
                )
            quart_app.config["dispatchers"].remove(sid)

            quart_app.config["dispatchers"].remove(sid)
        except Exception as e:
            logger.debug(f"Error {e}")

    @sio.event
    async def dataset_init(sid, data: str):
        """
        Handle dataset initialization event.
        1. Create data cache for a specific dataset.
        2. Create dispatcher and start it.
        :param data: Dictionary containing dataset information.
        :return:
        """
        logger.debug("event: dataset_init", sid=sid)

        # str to dict
        data = ast.literal_eval(data)

        dataset_name = data["dataset_name"]
        nfeat = data["nfeat"]
        nfield = data["nfield"]
        total_batch_num = data["nbatch"]
        cache_num = data["cache_num"]

        logger.info(f"[socket: on_dataset_init] on_dataset_init, receive: {data}")

        # 1. Create data cache if not exist
        data_cache: ContextStates[DataCache] = quart_app.config["data_cache"]
        if not data_cache.contains(sid, dataset_name):
            _cache = DataCache(
                dataset_name=dataset_name,
                total_batch_num=total_batch_num,
                maxsize=cache_num,
            )
            _cache.dataset_statistics = (nfeat, nfield)
            data_cache.add(sid, dataset_name, _cache)
        else:
            _cache = data_cache.get(sid, dataset_name)

        # 2. Create dispatcher if not exist
        dispatchers = quart_app.config["dispatchers"]
        if not dispatchers.contains(sid, dataset_name):
            _data_dispatcher = LibSvmDataDispatcher()
            dispatchers.add(sid, dataset_name, _data_dispatcher)
            _data_dispatcher.bound_client_to_cache(_cache, sid)
            _data_dispatcher.bound_sio(sio)

            loop = asyncio.get_event_loop()
            _data_dispatcher.bound_loop(loop)
            _data_dispatcher.start()
            
            t = Thread(target=_data_dispatcher.background_task)
            t.daemon = True
            t.start()

        await sio.emit("dataset_init", {"message": "Done"})

    @sio.event
    async def batch_data(sid, data: str):
        """
        Handle the event of receiving database data.
        Add the received data to the appropriate cache queue.
        :param data: Dictionary containing dataset information and the actual data.
        """
        logger.debug("event: batch_data", sid=sid)

        data = json.loads(data)

        dataset_name = data["dataset_name"]
        dataset = data["dataset"]

        logger.debug(
            f"[socket: batch_data]: {sid} receive_db_data name {dataset_name} and data {dataset[:10]}..."
        )

        # Check if dispatcher is launched for this dataset
        dispatchers = quart_app.config["dispatchers"]
        if not dispatchers.contains(sid, dataset_name):
            logger.debug(
                f"dispatchers is not initialized for dataset {dataset_name} and client {sid}, "
                f"wait for train/inference/finetune request"
            )
            await sio.emit(
                "response",
                {
                    "message": f"dispatchers is not initialized for dataset {dataset_name} and client {sid}, "
                    f"wait for train/inference/finetune request"
                },
            )
        else:
            # logger.debug(f"dispatchers is initialized for dataset {dataset_name} and client {socket_id}")
            dispatcher: LibSvmDataDispatcher = dispatchers.get(sid, dataset_name)
            if not dispatcher:
                logger.debug("[socket: on_batch_data]: dispatcher is not initialized")
                await sio.emit("response", {"message": "dispatcher is not initialized"})
            else:
                await dispatcher.add(dataset)
                await sio.emit(
                    "response", {"message": "Data received and added to queue!"}
                )

    return sio
