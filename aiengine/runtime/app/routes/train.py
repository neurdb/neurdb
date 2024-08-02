from flask import request, jsonify, current_app, g
from app.handlers.train import train
import traceback
from logger.logger import logger
from app.routes.routes import train_bp
from app.routes.context import before_execute
from cache import Bufferkey


@train_bp.route("/train", methods=["POST"])
def model_train():
    try:

        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        model_name = params.get("model_name")
        dataset_name = params.get("table_name")
        client_socket_id = params.get("client_socket_id")

        epoch = int(params.get("epoch"))
        train_batch_num = int(params.get("train_batch_num"))
        eva_batch_num = int(params.get("eva_batch_num"))
        test_batch_num = int(params.get("test_batch_num"))

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

        model_id = train(
            model_name=model_name,
            training_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            epochs=epoch,
            train_batch_num=train_batch_num,
            eva_batch_num=eva_batch_num,
            test_batch_num=test_batch_num,
        )
        logger.info(f"Training completed successfully with model_id: {model_id}")
        return jsonify({"model_id": model_id})

    except Exception as e:
        stacktrace = traceback.format_exc()
        error_message = {"res": "NA", "Errored": stacktrace}
        logger.error(f"model_train error: {str(e)}", stacktrace=stacktrace)
        return jsonify(error_message), 500
