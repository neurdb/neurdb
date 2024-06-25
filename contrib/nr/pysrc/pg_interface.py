import json
import traceback
import orjson
from argparse import Namespace
import calendar
import os
import time
import argparse
import examples.mlp as mlp
import configparser
from flask import Flask, request, jsonify

def parse_config_arguments(config_path: str):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    args = argparse.Namespace()

    args.log_folder = parser.get('DEFAULT', 'log_folder')
    args.log_name = parser.get('DEFAULT', 'log_name')
    args.base_dir = parser.get('DEFAULT', 'base_dir')
    args.model_repo = parser.get('DEFAULT', 'model_repo')

    # db config
    args.db_name = parser.get('DB_CONFIG', 'db_name')
    args.db_user = parser.get('DB_CONFIG', 'db_user')
    args.db_host = parser.get('DB_CONFIG', 'db_host')
    args.db_port = parser.get('DB_CONFIG', 'db_port')

    return args


def exception_catcher(func):
    def wrapper(encoded_str: str):
        global_res = "NA, "
        try:
            # each function accepts a json string
            params = json.loads(encoded_str)
            config_file = params.get("config_file")

            # Parse the config file
            args = parse_config_arguments(config_file)

            # Set the environment variables
            ts = calendar.timegm(time.gmtime())
            os.environ.setdefault("base_dir", args.base_dir)
            os.environ.setdefault("log_logger_folder_name", args.log_folder)
            os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")

            # Call the original function with the parsed parameters
            global_res = func(params, args)
            return global_res
        except Exception as e:
            return orjson.dumps(
                {"res": global_res, "Errored": traceback.format_exc()}).decode('utf-8')

    return wrapper


@exception_catcher
def mlp_clf(params: dict, args: Namespace):
    where_cond = params.get("where_cond")
    table = params.get("table")
    label = params.get("label")
    model_repo = args.model_repo
    result = mlp.run(table, where_cond, label, model_repo)
    return orjson.dumps({"res": result}).decode('utf-8')


@exception_catcher
def measure_call_overheads(params: dict, args: Namespace):
    return orjson.dumps({"Done": 1}).decode('utf-8')


import numpy as np
from multiprocessing import shared_memory


def get_data_from_shared_memory_int(n_rows):
    shm = shared_memory.SharedMemory(name="my_shared_memory")
    data = np.frombuffer(shm.buf, dtype=np.float32)
    data = data.reshape(n_rows, -1)
    return data

# Initialize the Flask application
app = Flask(__name__)

# Define the /mlp_clf route
@app.route('/mlp_clf', methods=['POST'])
def handle_mlp_clf():
    data = request.get_json()
    params_json = json.dumps(data)
    result = mlp_clf(params_json)
    return jsonify(result)

if __name__ == "__main__":
    # Start the Flask server
    app.run(host='localhost', port=8090)
