import asyncio
from quart import Quart, current_app, jsonify, g
from quart import request
from sio import create_sio_app
from cache import LibSvmDataDispatcher, DataCache, ContextStates
from connection.nr import NeurDBModelHandler
from shared_config.config import parse_config_arguments
from logger.logger import logger
from cache import Bufferkey
from app.routes.context import before_execute
from app.handlers.train import train
from app.handlers.inference import inference
import traceback
from logger.logger import configure_logging
from neurdb.logger import configure_logging as api_configure_logging
import socketio


configure_logging(None)
api_configure_logging(None)


MSG_NO_PARAMS = "No params"


quart_app = Quart(__name__)
sio = create_sio_app(quart_app)
sio_app = socketio.ASGIApp(sio, other_asgi_app=quart_app)

config_path = "./config.ini"
config_args = parse_config_arguments(config_path)
print(config_args)


if config_args.run_model != "in_database":
    NEURDB_CONNECTOR = None
else:
    NEURDB_CONNECTOR = NeurDBModelHandler(
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


@quart_app.route("/train", methods=["POST"])
async def model_train():
    try:
        # Use request.form to get form data
        params = await request.form
        if not params:
            return {"message": MSG_NO_PARAMS}, 400

        batch_size = params.get("batch_size", type=int)
        model_name = params.get("model_name")
        dataset_name = params.get("table_name")
        client_socket_id = params.get("client_socket_id")

        epoch = params.get("epoch", type=int)
        train_batch_num = params.get("train_batch_num", type=int)
        eva_batch_num = params.get("eva_batch_num", type=int)
        test_batch_num = params.get("test_batch_num", type=int)

        features = params.get("features")
        target = params.get("target")  # always one target

        config_args = current_app.config["config_args"]
        db_connector = current_app.config["db_connector"]

        logger.info(f"[model_train]: receive params {params}")

        exe_flag, exe_info = before_execute(
            dataset_name=dataset_name,
            data_key=Bufferkey.TRAIN_KEY,
            client_id=client_socket_id,
        )
        if not exe_flag:
            logger.error(f"Execution flag failed: {exe_info}")
            return jsonify(exe_info), 400

        model_id = await train(
            model_name=model_name,
            training_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            epochs=epoch,
            train_batch_num=train_batch_num,
            eva_batch_num=eva_batch_num,
            test_batch_num=test_batch_num,
            table_name=dataset_name,
            features=features.split(","),
            target=target,
        )
        logger.info(f"Training completed successfully with model_id: {model_id}")
        return jsonify({"model_id": model_id})

    except Exception as e:
        stacktrace = traceback.format_exc()
        error_message = {"res": "NA", "Errored": stacktrace}
        logger.error(f"model_train error: {str(e)}", stacktrace=stacktrace)
        return jsonify(error_message), 500


@quart_app.route("/inference", methods=["POST"])
async def model_inference():
    try:
        params = await request.form
        if not params:
            return {"message": MSG_NO_PARAMS}, 400

        model_name = params.get("model_name")
        model_id = params.get("model_id", type=int)
        batch_size = params.get("batch_size", type=int)
        dataset_name = params.get("table_name")
        client_socket_id = params.get("client_socket_id")

        inf_batch_num = params.get("batch_num", type=int)

        config_args = current_app.config["config_args"]
        db_connector = current_app.config["db_connector"]

        exe_flag, exe_info = before_execute(
            dataset_name=dataset_name,
            data_key=Bufferkey.INFERENCE_KEY,
            client_id=client_socket_id,
        )
        if not exe_flag:
            return jsonify(exe_info), 400

        result = await inference(
            model_name=model_name,
            inference_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            model_id=model_id,
            inf_batch_num=inf_batch_num,
        )

        # todo: make the response as result
        logger.debug("---- Inference return to UDF ---- ")
        logger.info(
            f"---- Inference done for {len(result) * len(result[0])} samples ----"
        )
        return jsonify({"res": "Done"})

    except Exception:
        stacktrace = traceback.format_exc()
        error_message = {"res": "NA", "Errored": stacktrace}
        logger.error("model_inference error", stacktrace=stacktrace)
        return jsonify(error_message), 500


app = sio_app
