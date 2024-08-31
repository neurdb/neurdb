import asyncio
import json
import uuid
from threading import Thread

from neurdb.logger import configure_logging as api_configure_logging
from neurdbrt.app.handlers.finetune import finetune
from neurdbrt.app.handlers.inference import inference
from neurdbrt.app.handlers.train import train
from neurdbrt.app.hook import before_execute
from neurdbrt.cache import Bufferkey, ContextStates, DataCache, LibSvmDataDispatcher
from neurdbrt.config import parse_config_arguments
from neurdbrt.log import configure_logging, logger
from neurdbrt.repo import ModelRepository
from neurdbrt.websocket_sender import WebsocketSender
from quart import Quart, current_app, g, websocket

configure_logging(None)
api_configure_logging(None)

MSG_NO_PARAMS = "No params"

quart_app = Quart(__name__)

config_path = "./config.ini"
config_args = parse_config_arguments(config_path)
print(config_args)

if config_args.run_model != "in_database":
    NEURDB_CONNECTOR = None
else:
    NEURDB_CONNECTOR = ModelRepository(
        {
            "db_name": config_args.db_name,
            "db_user": config_args.db_user,
            "db_host": config_args.db_host,
            "db_port": config_args.db_port,
            # "password": config_args.db_password,
        }
    )


@quart_app.before_serving
async def before_serving_func():
    logger.info("before_serving_func")

    # shared global contexts among tasks.
    async with quart_app.app_context():
        quart_app.config["config_args"] = config_args
        quart_app.config["db_connector"] = NEURDB_CONNECTOR
        quart_app.config["data_cache"] = ContextStates[DataCache]()
        quart_app.config["dispatchers"] = ContextStates[LibSvmDataDispatcher]()
        quart_app.config["clients"] = {}


@quart_app.route("/")
async def hello():
    return "This is NeurDB Python Server."


@quart_app.websocket("/ws")
async def handle_ws():
    sender_task = asyncio.create_task(WebsocketSender.start_websocket_sender_task())
    logger.debug(f"Client event received.")
    while True:
        data = await websocket.receive()
        if data:
            data_json: dict = json.loads(data)
            event = data_json.get("event")
            if event == "setup":
                await on_setup(data_json)
            elif event == "disconnect":
                await on_disconnect(data_json)
            elif event == "task":
                await on_task(data_json)
            elif event == "batch_data":
                await on_batch_data(data_json)
            elif event == "ack_result":
                await on_ack_result(data_json)
            else:
                pass


async def on_setup(data: dict):
    logger.debug(f"Client connected: {data}")
    session_id = str(uuid.uuid4())
    quart_app.config["clients"][
        session_id
    ] = session_id  # TODO: refactor the structure of clients map
    await WebsocketSender.send(
        json.dumps({"version": 1, "event": "ack_setup", "sessionId": session_id})
    )


async def on_disconnect(data: dict):
    logger.debug(f"Client disconnected: {data}")
    session_id = data.get("sessionId")
    quart_app.config["clients"].pop(session_id)
    quart_app.config["data_cache"].remove(session_id)
    quart_app.config["dispatchers"].remove(session_id)
    await WebsocketSender.send(
        json.dumps({"version": 1, "event": "ack_disconnect", "sessionId": session_id})
    )
    WebsocketSender.stop()


async def on_task(data: dict):
    logger.debug(f"Task received: {data}")
    session_id = data.get("sessionId")
    task: str = data.get("type")
    if task == "train":
        await on_train(data)
    elif task == "inference":
        await on_inference(data)
    elif task == "finetune":
        await on_finetune(data)
    else:
        pass


async def on_batch_data(data: dict):
    logger.debug(f"Batch data received")
    session_id = data.get("sessionId")
    if not quart_app.config["data_cache"].contains(session_id, session_id):
        logger.error(
            f"Data cache is not initialized for client {session_id} but received batch data"
        )
    else:
        data = data["byte"]
        dispatcher = quart_app.config["dispatchers"].get(session_id, session_id)
        await dispatcher.add(data)


async def on_ack_result(data: dict):
    logger.debug(f"Ack result received")


async def on_train(data: dict):
    n_feat = data["nFeat"]
    n_field = data["nField"]
    total_batch_num = (
        data["spec"]["nBatchTrain"]
        + data["spec"]["nBatchEval"]
        + data["spec"]["nBatchTest"]
    )
    cache_size = data["cacheSize"]
    session_id = data["sessionId"]
    await init_database(n_feat, n_field, total_batch_num, cache_size, session_id)
    await websocket.send(
        json.dumps({"version": 1, "event": "ack_task", "sessionId": session_id})
    )

    exe_flag, exe_info = before_execute(
        dataset_name=session_id,
        data_key=Bufferkey.TRAIN_KEY,
        client_id=session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    train_task = asyncio.create_task(
        train(
            model_name=data["architecture"],
            training_libsvm=g.data_loader,
            args=current_app.config["config_args"],
            db=current_app.config["db_connector"],
            epoch=data["spec"]["epoch"],
            train_batch_num=data["spec"]["nBatchTrain"],
            eval_batch_num=data["spec"]["nBatchEval"],
            test_batch_num=data["spec"]["nBatchTest"],
            features=data["features"],
            target=data["target"],
        )
    )

    train_task.add_done_callback(lambda task: task_done_callback(task, session_id))


def task_done_callback(task, session_id):
    asyncio.create_task(
        WebsocketSender.send(
            json.dumps(
                {
                    "version": 1,
                    "event": "result",
                    "sessionId": session_id,
                    "payload": "Task completed",
                }
            )
        )
    )


async def on_inference(data: dict):
    n_feat = data["nFeat"]
    n_field = data["nField"]
    total_batch_num = data["spec"]["nBatch"]
    cache_size = data["cacheSize"]
    session_id = data["sessionId"]
    await init_database(n_feat, n_field, total_batch_num, cache_size, session_id)
    await websocket.send(
        json.dumps({"version": 1, "event": "ack_task", "sessionId": session_id})
    )

    exe_flag, exe_info = before_execute(
        dataset_name=session_id,
        data_key=Bufferkey.INFERENCE_KEY,
        client_id=session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    inference_task = asyncio.create_task(
        inference(
            model_name=data["architecture"],
            inference_libsvm=g.data_loader,
            args=current_app.config["config_args"],
            db=current_app.config["db_connector"],
            model_id=data["modelId"],
            inf_batch_num=total_batch_num,
        )
    )

    inference_task.add_done_callback(lambda task: task_done_callback(task, session_id))


async def on_finetune(data: dict):
    n_feat = data["nFeat"]
    n_field = data["nField"]
    total_batch_num = (
        data["spec"]["nBatchTrain"]
        + data["spec"]["nBatchEval"]
        + data["spec"]["nBatchTest"]
    )
    cache_size = data["cacheSize"]
    session_id = data["sessionId"]
    await init_database(n_feat, n_field, total_batch_num, cache_size, session_id)
    await websocket.send(
        json.dumps({"version": 1, "event": "ack_task", "sessionId": session_id})
    )

    exe_flag, exe_info = before_execute(
        dataset_name=session_id,
        data_key=Bufferkey.TRAIN_KEY,
        client_id=session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    finetune_task = asyncio.create_task(
        finetune(
            model_name=data["architecture"],
            finetune_libsvm=g.data_loader,
            args=current_app.config["config_args"],
            db=current_app.config["db_connector"],
            model_id=data["modelId"],
            epoch=data["spec"]["epoch"],
            train_batch_num=data["spec"]["nBatchTrain"],
            eva_batch_num=data["spec"]["nBatchEval"],
            test_batch_num=data["spec"]["nBatchTest"],
        )
    )

    finetune_task.add_done_callback(lambda task: task_done_callback(task, session_id))


async def init_database(
    n_feat: int, n_field: int, total_batch_num: int, cache_size: int, session_id: str
):
    data_cache = quart_app.config["data_cache"]
    if not data_cache.contains(session_id, session_id):
        _cache = DataCache(
            dataset_name=session_id,
            total_batch_num=total_batch_num,
            maxsize=cache_size,
        )
        _cache.dataset_statistics = (n_feat, n_field)
        data_cache.add(session_id, session_id, _cache)
    else:
        _cache = data_cache.get(session_id, session_id)

    dispatchers = quart_app.config["dispatchers"]
    if not dispatchers.contains(session_id, session_id):
        _data_dispatcher = LibSvmDataDispatcher()
        dispatchers.add(session_id, session_id, _data_dispatcher)
        _data_dispatcher.bound_client_to_cache(_cache, session_id)
        _data_dispatcher.start()
        loop = asyncio.get_event_loop()
        _data_dispatcher.bound_loop(loop)
        _data_dispatcher.start()
        t = Thread(target=_data_dispatcher.background_task)
        t.daemon = True
        t.start()


app = quart_app
