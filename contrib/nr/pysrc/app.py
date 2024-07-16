from logger.logger import logger
import traceback
import orjson
from flask import Flask, request, jsonify, g
from shared_config.config import parse_config_arguments
from apps import build_model
from connection.pg_connect import DatabaseModelHandler
from utils.dataset import libsvm_dataloader, build_inference_loader
from utils.io import save_model_weight, load_model_weight
from io import BytesIO
from cache.model_cache import ModelCache

app = Flask(__name__)

# Load config and initialize once
config_args = parse_config_arguments("./config.ini")
model_cache = ModelCache()

db_connector = DatabaseModelHandler({
    'dbname': config_args.db_name,
    'user': config_args.db_user,
    'host': config_args.db_host,
    'port': config_args.db_port,
    'password': config_args.db_password,
})


# db_connector.connect_to_db()
# db_connector.create_model_table()


@app.before_request
def before_request():
    g.config_args = config_args
    g.db_connector = db_connector
    g.model_cache = model_cache


@app.route('/train', methods=['POST'])
def model_train():
    try:
        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        libsvm_file = request.files['libsvm_file']
        model_name = params.get("model_name")

        file_obj = BytesIO(libsvm_file.read())

        train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
            batch_size, g.config_args.data_loader_worker, file_obj)

        builder = build_model(model_name, g.config_args)
        builder.model_dimension = (nfeat, nfields)
        builder.train(train_loader, val_loader, test_loader)

        model_binary, model_path = save_model_weight(builder.model, g.config_args.model_repo)
        # model_id = g.db_connector.insert_model_binary(model_path, model_name, model_binary)
        model_id = g.model_cache.add_model(model_name, builder)

        return jsonify({"model_id": model_id})

    except Exception as e:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        print(traceback.format_exc())
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500


@app.route('/inference', methods=['POST'])
def model_inference():
    try:
        params = request.form  # Use requ
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_file = request.files['libsvm_file']

        file_obj = BytesIO(libsvm_file.read())

        inference_loader, nfields, nfeat = build_inference_loader(
            g.config_args.data_loader_worker, file_obj)

        # load model
        builder = g.model_cache.get_model(model_name, model_id)
        if builder is None:
            model_name, model_binary = g.db_connector.get_model_binary(model_id)
            if model_binary:
                builder = build_model(model_name, g.config_args)
                load_model_weight(model_binary, builder.model)

        if not builder:
            return jsonify({"res": "Model Not trained yet"}), 404

        # check if test data matching model dimension
        model_nfeat, model_nfields = builder.model_dimension
        if nfields > model_nfields or nfeat > model_nfeat:
            return jsonify({
                "res": (
                    f"Model trained with nfields = {model_nfields} and nfeat = {model_nfeat} "
                    f"cannot handle input with nfields = {nfields} and nfeat = {nfeat}."
                )
            }), 404

        infer_res = builder.inference(inference_loader)

        return jsonify({"res": infer_res})

    except Exception as e:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config_args.server_port)
