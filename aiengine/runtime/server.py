import asyncio
import json
from threading import Thread
from typing import Any, List

from neurdb.logger import configure_logging as api_configure_logging
from neurdbrt.app import Setup, WebsocketSender, before_execute
from neurdbrt.app.msg import (
    AckTaskResponse,
    DisconnectRequest,
    DisconnectResponse,
    InferenceResultResponse,
    ModelResultResponse,
    SetupRequest,
    SetupResponse,
    TaskRequest,
)
from neurdbrt.cache import Bufferkey, ContextStates, DataCache, LibSvmDataDispatcher
from neurdbrt.config import parse_config_arguments
from neurdbrt.hook import exec_hooks_on_start, register_hooks
from neurdbrt.log import configure_logging, logger
from neurdbrt.repo import ModelRepository
from quart import Quart, current_app, g, websocket

configure_logging(None)
api_configure_logging(None)

MSG_NO_PARAMS = "No params"

quart_app = Quart(__name__)

config_path = "./config.ini"
config_args = parse_config_arguments(config_path)
logger.info(config_args)

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

register_hooks()
exec_hooks_on_start()


def inject_task_args():
    # TODO: receive from DB engine
    current_app.config["config_args"].noutput = 2


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

    req = SetupRequest(data)
    quart_app.config["clients"][
        req.session_id
    ] = req.session_id  # TODO: refactor the structure of clients map

    await WebsocketSender.send(SetupResponse(req.session_id).to_json())


async def on_disconnect(data: dict):
    logger.debug(f"Client disconnected: {data}")

    req = DisconnectRequest(data)
    quart_app.config["clients"].pop(req.session_id)
    quart_app.config["data_cache"].remove(req.session_id)
    quart_app.config["dispatchers"].remove(req.session_id)

    await WebsocketSender.send(DisconnectResponse(req.session_id).to_json())
    WebsocketSender.stop()


async def on_task(data: dict):
    logger.debug(f"Task received: {data}")
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
    req = TaskRequest(data, is_inference=False)
    await init_database(req, in_libsvm_format=(data["architecture"] != "auto_pipeline"))

    await websocket.send(AckTaskResponse(req.session_id).to_json())

    exe_flag, exe_info = before_execute(
        dataset_name=req.session_id,
        data_key=Bufferkey.TRAIN_KEY,
        client_id=req.session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    inject_task_args()

    asyncio.create_task(
        train_task(
            setup=Setup(
                model_name=data["architecture"],
                libsvm_data=g.data_loader,
                args=current_app.config["config_args"],
                db=current_app.config["db_connector"],
            ),
            table_name=data["table"],
            epoch=data["spec"]["epoch"],
            train_batch_num=data["spec"]["nBatchTrain"],
            eval_batch_num=data["spec"]["nBatchEval"],
            test_batch_num=data["spec"]["nBatchTest"],
            features=data["features"],
            target=data["target"],
        )
    ).add_done_callback(lambda task: train_done_callback(task.result(), req.session_id))


def train_done_callback(model_id, session_id):
    asyncio.create_task(
        WebsocketSender.send(ModelResultResponse(session_id, model_id).to_json())
    )


async def train_task(
    setup: Setup,
    table_name: str,
    epoch: int,
    train_batch_num: int,
    eval_batch_num: int,
    test_batch_num: int,
    features: str,
    target: str,
) -> int:
    logger.info(
        f"train_task called",
        table_name=table_name,
        epoch=epoch,
        train_batch_num=train_batch_num,
        eval_batch_num=eval_batch_num,
        test_batch_num=test_batch_num,
        features=features,
        target=target,
    )

    model_id, err = await setup.train(
        epoch, train_batch_num, eval_batch_num, test_batch_num
    )
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"train done. model_id: {model_id}")

    if NEURDB_CONNECTOR:
        NEURDB_CONNECTOR.register_model(model_id, table_name, features, target)

    return model_id


async def on_inference(data: dict):
    req = TaskRequest(data, is_inference=True)

    await init_database(req, in_libsvm_format=(data["architecture"] != "auto_pipeline"))
    await websocket.send(AckTaskResponse(req.session_id).to_json())

    exe_flag, exe_info = before_execute(
        dataset_name=req.session_id,
        data_key=Bufferkey.INFERENCE_KEY,
        client_id=req.session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    inject_task_args()

    asyncio.create_task(
        inference_task(
            setup=Setup(
                model_name=data["architecture"],
                libsvm_data=g.data_loader,
                args=current_app.config["config_args"],
                db=current_app.config["db_connector"],
            ),
            model_id=data["modelId"],
            inf_batch_num=req.total_batch_num,
            table_name=data["table"],
            features=data["features"].split(","),
            target=data["target"],
        )
    ).add_done_callback(
        lambda task: inference_done_callback(task.result(), req.session_id)
    )


def inference_done_callback(result, session_id):
    asyncio.create_task(
        WebsocketSender.send(InferenceResultResponse(session_id, result).to_json())
    )


async def inference_task(
    setup: Setup,
    model_id: int,
    table_name: str,
    inf_batch_num: int,
    features: List[str],
    target: str,
) -> List[List[Any]]:
    logger.info(
        f"train_task called",
        table_name=table_name,
        inf_batch_num=inf_batch_num,
        features=features,
        target=target,
    )

    response, err = await setup.inference(model_id, inf_batch_num, features, target)
    if err is not None:
        logger.error(f"inference failed with error: {err}")
        return []

    logger.debug(f"inference done. response")

    return response


async def on_finetune(data: dict):
    req = TaskRequest(data, is_inference=False)

    await init_database(req, in_libsvm_format=(data["architecture"] != "auto_pipeline"))
    await websocket.send(AckTaskResponse(req.session_id).to_json())

    exe_flag, exe_info = before_execute(
        dataset_name=req.session_id,
        data_key=Bufferkey.TRAIN_KEY,
        client_id=req.session_id,
    )
    if not exe_flag:
        logger.error(f"Execution flag failed: {exe_info}")
        return

    inject_task_args()

    asyncio.create_task(
        finetune_task(
            setup=Setup(
                model_name=data["architecture"],
                libsvm_data=g.data_loader,
                args=current_app.config["config_args"],
                db=current_app.config["db_connector"],
            ),
            model_id=data["modelId"],
            epoch=data["spec"]["epoch"],
            train_batch_num=data["spec"]["nBatchTrain"],
            eva_batch_num=data["spec"]["nBatchEval"],
            test_batch_num=data["spec"]["nBatchTest"],
        )
    ).add_done_callback(
        lambda task: finetune_done_callback(task.result(), req.session_id)
    )


def finetune_done_callback(model_id, session_id):
    asyncio.create_task(
        WebsocketSender.send(ModelResultResponse(session_id, model_id).to_json())
    )


async def finetune_task(
    setup: Setup,
    model_id: int,
    epoch: int,
    train_batch_num: int,
    eva_batch_num: int,
    test_batch_num: int,
) -> int:
    model_id, err = await setup.finetune(
        model_id,
        start_layer_id=5,
        epoch=epoch,
        train_batch_num=train_batch_num,
        eva_batch_num=eva_batch_num,
        test_batch_num=test_batch_num,
    )
    if err is not None:
        logger.error(f"train failed with error: {err}")
        return -1

    print(f"finetune done. model_id: {model_id}")
    return model_id


async def init_database(req: TaskRequest, in_libsvm_format: bool = True):
    # Get the arguments from the request
    session_id = req.session_id
    n_feat = req.n_feat
    n_field = req.n_field
    total_batch_num = req.total_batch_num
    cache_size = req.cache_size

    # Create the data cache if it doesn't exist
    cache = quart_app.config["data_cache"]
    if not cache.contains(session_id, session_id):
        c = DataCache(
            dataset_name=session_id, total_batch_num=total_batch_num, maxsize=cache_size
        )
        c.dataset_statistics = (n_feat, n_field)
        cache.add(session_id, session_id, c)
    else:
        c = cache.get(session_id, session_id)

    # Create the data dispatcher if it doesn't exist
    dispatchers = quart_app.config["dispatchers"]
    if not dispatchers.contains(session_id, session_id):
        d = LibSvmDataDispatcher(in_libsvm_format=in_libsvm_format)
        logger.debug(f"Created data dispatcher", in_libsvm_format=in_libsvm_format)
        dispatchers.add(session_id, session_id, d)

        d.bound_client_to_cache(c, session_id)
        d.start()

        loop = asyncio.get_event_loop()
        d.bound_loop(loop)
        d.start()

        t = Thread(target=d.background_task)
        t.daemon = True
        t.start()


app = quart_app
