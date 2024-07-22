from flask import request, jsonify, current_app
from app.handlers.finetune import finetune
import traceback
import orjson
from logger.logger import logger
from app.routes.blueprints import finetune_bp


@finetune_bp.route('/finetune', methods=['POST'])
def model_finetune():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_data = params.get("libsvm_data")
        batch_size = int(params.get("batch_size"))

        config_args = current_app.config['config_args']
        db_connector = current_app.config['db_connector']

        model_id = finetune(
            model_name=model_name,
            finetune_libsvm=libsvm_data,
            args=config_args,
            db=db_connector,
            model_id=model_id,
            batch_size=batch_size,
        )
        return jsonify({"model_id": model_id})

    except Exception:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500
