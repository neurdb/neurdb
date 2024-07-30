from flask import request, jsonify, current_app, g
from app.handlers.inference import inference
import traceback
from logger.logger import logger
from app.routes.routes import inference_bp
from app.routes.context import before_execute
from cache import Bufferkey


@inference_bp.route("/inference", methods=["POST"])
def model_inference():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        batch_size = int(params.get("batch_size"))
        dataset_name = params.get("table_name")
        client_socket_id = params.get("client_socket_id")

        inf_batch_num = int(params.get("batch_num"))

        config_args = current_app.config["config_args"]
        db_connector = current_app.config["db_connector"]

        exe_flag, exe_info = before_execute(
            dataset_name=dataset_name,
            data_key=Bufferkey.INFERENCE_KEY,
            client_id=client_socket_id,
        )
        if not exe_flag:
            return jsonify(exe_info), 400

        result = inference(
            model_name=model_name,
            inference_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            model_id=model_id,
            inf_batch_num=inf_batch_num,
        )

        # todo: make the response as result
        logger.debug("---- Inference return to UDF ---- ")
        logger.info(f"---- Inference done for {len(result) * len(result[0])} samples ----")
        return jsonify({"res": "Done"})

    except Exception:
        stacktrace = traceback.format_exc()
        error_message = {"res": "NA", "Errored": stacktrace}
        logger.error("model_inference error", stacktrace=stacktrace)
        return jsonify(error_message), 500
