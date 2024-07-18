import os

from connection import NeurDBModelHandler
from logger.logger import logger, configure_logging
import traceback
import orjson
from flask import Flask, request, jsonify, g
from shared_config.config import parse_config_arguments
from cache.model_cache import ModelCache

from cli import train, inference, finetune


configure_logging("./app.log")

app = Flask(__name__)

# Load config and initialize once
config_path = "./config.ini"
# TODO: hardcoded path here, need to be fixed in the future
if os.path.exists('$NEURDBPATH/contrib/nr/pysrc/config.ini'):
    config_path = os.path.expandvars('$NEURDBPATH/contrib/nr/pysrc/config.ini')

config_args = parse_config_arguments(config_path)
model_cache = ModelCache()

# db_connector = DatabaseModelHandler({
#     'dbname': config_args.db_name,
#     'user': config_args.db_user,
#     'host': config_args.db_host,
#     'port': config_args.db_port,
#     'password': config_args.db_password,
# })


NEURDB_CONNECTOR = NeurDBModelHandler(
    {
        "db_name": config_args.db_name,
        "db_user": config_args.db_user,
        "db_host": config_args.db_host,
        "db_port": config_args.db_port,
        # "password": config_args.db_password,
    }
)

# db_connector.connect_to_db()
# db_connector.create_model_table()


@app.before_request
def before_request():
    g.config_args = config_args
    g.db_connector = NEURDB_CONNECTOR
    g.model_cache = model_cache


@app.route('/train', methods=['POST'])
def model_train():
    try:
        params = request.form  # Use request.form to get form data
        batch_size = int(params.get("batch_size"))
        model_name = params.get("model_name")
        data = params.get("libsvm_data")



        model_id = train(
            model_name=model_name,
            training_libsvm=data,
            args=config_args,
            db=NEURDB_CONNECTOR,
            batch_size=batch_size
        )

        # train_loader, val_loader, test_loader, nfields, nfeat = libsvm_dataloader(
        #     batch_size, g.config_args.data_loader_worker, data)
        #
        # builder = build_model(model_name, g.config_args)
        # builder.model_dimension = (nfeat, nfields)
        # builder.train(train_loader, val_loader, test_loader)
        #
        # model_binary, model_path = save_model_weight(builder.model, g.config_args.model_repo)
        # # model_id = g.db_connector.insert_model_binary(model_path, model_name, model_binary)
        # model_id = g.model_cache.add_model(model_name, builder)

        return jsonify({"model_id": model_id})

    except Exception:
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
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_data = params.get("libsvm_data")
        batch_size = int(params.get("batch_size"))

        result = inference(
            model_name=model_name,
            inference_libsvm=libsvm_data,
            args=config_args,
            db=NEURDB_CONNECTOR,
            model_id=model_id,
            batch_size=batch_size
        )

        # inference_loader, nfields, nfeat = build_inference_loader(
        #     g.config_args.data_loader_worker, libsvm_data, batch_size)
        #
        # # load model
        # builder = g.model_cache.get_model(model_name, model_id)
        # if builder is None:
        #     model_name, model_binary = g.db_connector.get_model_binary(model_id)
        #     if model_binary:
        #         builder = build_model(model_name, g.config_args)
        #         load_model_weight(model_binary, builder.model)
        #
        # if not builder:
        #     return jsonify({"res": "Model Not trained yet"}), 404
        #
        # # check if test data matching model dimension
        # model_nfeat, model_nfields = builder.model_dimension
        # if nfields > model_nfields or nfeat > model_nfeat:
        #     return jsonify({
        #         "res": (
        #             f"Model trained with nfields = {model_nfields} and nfeat = {model_nfeat} "
        #             f"cannot handle input with nfields = {nfields} and nfeat = {nfeat}."
        #         )
        #     }), 404
        #
        # infer_res = builder.inference(inference_loader)

        return jsonify({"res": result})

    except Exception:
        error_message = {
            "res": "NA",
            "Errored": traceback.format_exc()
        }
        logger.error(orjson.dumps(error_message).decode('utf-8'))
        return jsonify(error_message), 500


@app.route('/finetune', methods=['POST'])
def model_finetune():
    try:
        params = request.form  # Use request.form to get form data
        model_name = params.get("model_name")
        model_id = int(params.get("model_id"))
        libsvm_data = params.get("libsvm_data")
        batch_size = int(params.get("batch_size"))

        model_id = finetune(
            model_name=model_name,
            finetune_libsvm=libsvm_data,
            args=config_args,
            db=NEURDB_CONNECTOR,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config_args.server_port)
