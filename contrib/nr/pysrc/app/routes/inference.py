from flask import request, jsonify, g
from app.handlers.inference import inference
import traceback
import orjson
from logger.logger import logger
from . import inference_bp


@inference_bp.route('/inference', methods=['POST'])
def model_inference():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_data = params.get("libsvm_data")
        batch_size = int(params.get("batch_size"))

        result = inference(
            model_name=model_name,
            inference_libsvm=libsvm_data,
            args=g.config_args,
            db=g.db_connector,
            model_id=model_id,
            batch_size=batch_size
        )

        return jsonify({"res": result})

    except Exception:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500
