from flask import request, jsonify, current_app
from app.handlers.train import train
import traceback
import orjson
from logger.logger import logger
from app.routes.blueprints import train_bp


@train_bp.route('/train', methods=['POST'])
def model_train():
    try:
        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        model_name = params.get("model_name")
        data = params.get("libsvm_data")

        config_args = current_app.config['config_args']
        db_connector = current_app.config['db_connector']

        model_id = train(
            model_name=model_name,
            training_libsvm=data,
            args=config_args,
            db=db_connector,
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
