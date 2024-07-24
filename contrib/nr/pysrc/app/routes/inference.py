from flask import request, jsonify, current_app, g
from app.handlers.inference import inference
import traceback
import orjson
from logger.logger import logger
from app.routes.blueprints import inference_bp
from app.routes.context import before_execute
from cache.data_cache import Bufferkey


@inference_bp.route('/inference', methods=['POST'])
def model_inference():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_data = params.get("libsvm_data")
        batch_size = int(params.get("batch_size"))
        dataset_name = params.get("dataset_name")
        client_socket_id = params.get("client_socket_id")

        inf_batch_num = int(params.get("inf_batch_num"))

        config_args = current_app.config['config_args']
        db_connector = current_app.config['db_connector']

        exe_flag, exe_info = before_execute(dataset_name=dataset_name, data_key=Bufferkey.INFERENCE_KEY,
                                            client_id=client_socket_id)
        if not exe_flag:
            return jsonify(exe_info), 400

        result = inference(
            model_name=model_name,
            inference_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            model_id=model_id,
            batch_size=batch_size,
            inf_batch_num=inf_batch_num
        )

        return jsonify({"res": result})

    except Exception:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500
