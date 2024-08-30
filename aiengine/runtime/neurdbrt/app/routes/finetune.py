import traceback

from neurdbrt.app.handlers.finetune import finetune
from neurdbrt.app.routes.context import before_execute
from neurdbrt.app.routes.routes import finetune_bp
from neurdbrt.cache import Bufferkey
from flask import current_app, g, jsonify, request
from neurdbrt.log import logger


@finetune_bp.route("/finetune", methods=["POST"])
def model_finetune():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        batch_size = int(params.get("batch_size"))
        dataset_name = params.get("table_name")
        client_socket_id = params.get("client_socket_id")

        epoch = int(params.get("epoch"))
        train_batch_num = int(params.get("train_batch_num"))
        eva_batch_num = int(params.get("eva_batch_num"))
        test_batch_num = int(params.get("test_batch_num"))

        config_args = current_app.config["config_args"]
        db_connector = current_app.config["db_connector"]

        exe_flag, exe_info = before_execute(
            dataset_name=dataset_name,
            data_key=Bufferkey.TRAIN_KEY,
            client_id=client_socket_id,
        )
        if not exe_flag:
            return jsonify(exe_info), 400

        model_id = finetune(
            model_name=model_name,
            finetune_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            model_id=model_id,
            epoch=epoch,
            train_batch_num=train_batch_num,
            eva_batch_num=eva_batch_num,
            test_batch_num=test_batch_num,
        )
        return jsonify({"model_id": model_id})

    except Exception:
        stacktrace = traceback.format_exc()
        error_message = {"res": "NA", "Errored": stacktrace}
        logger.error("model_finetune error", stacktrace=stacktrace)
        return jsonify(error_message), 500
