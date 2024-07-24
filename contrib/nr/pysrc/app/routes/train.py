from flask import request, jsonify, current_app, g
from app.handlers.train import train
import traceback
import orjson
from logger.logger import logger
from app.routes.blueprints import train_bp
from app.routes.context import before_execute
from cache.data_cache import Bufferkey


@train_bp.route('/train', methods=['POST'])
def model_train():
    try:
        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        model_name = params.get("model_name")
        data = params.get("libsvm_data")
        dataset_name = params.get("dataset_name")
        client_socket_id = params.get("client_socket_id")

        epoch = int(params.get("epoch"))
        train_batch_num = int(params.get("train_batch_num"))
        eva_batch_num = int(params.get("eva_batch_num"))
        test_batch_num = int(params.get("test_batch_num"))

        config_args = current_app.config['config_args']
        db_connector = current_app.config['db_connector']

        exe_flag, exe_info = before_execute(dataset_name=dataset_name, data_key=Bufferkey.TRAIN_KEY,
                                            client_id=client_socket_id)
        if not exe_flag:
            return jsonify(exe_info), 400

        model_id = train(
            model_name=model_name,
            training_libsvm=g.data_loader,
            args=config_args,
            db=db_connector,
            batch_size=batch_size,
            epochs=epoch,
            train_batch_num=train_batch_num,
            eva_batch_num=eva_batch_num,
            test_batch_num=test_batch_num
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
