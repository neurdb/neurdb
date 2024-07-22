from flask import request, jsonify, g
from app.handlers.train import train
import traceback
import orjson
from logger.logger import logger
from . import train_bp


@train_bp.route('/train', methods=['POST'])
def model_train():
    try:
        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        model_name = params.get("model_name")
        data = params.get("libsvm_data")

        model_id = train(
            model_name=model_name,
            training_libsvm=data,
            args=g.config_args,
            db=g.db_connector,
            batch_size=batch_size
        )

        return jsonify({"model_id": model_id})

    except Exception:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        print(traceback.format_exc())
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500
