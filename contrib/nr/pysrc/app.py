from logger.logger import configure_logging
from flask import Flask
from shared_config.config import parse_config_arguments
from app.routes import train_bp, inference_bp, finetune_bp
from app.routes.context import before_request_func, after_request_func
from app.socketio.data_socketio import socketio, NRDataManager
from cache import ContextStates, DataCache, LibSvmDataDispatcher
from connection import NeurDBModelHandler
from neurdb.logger import configure_logging as api_configure_logging
from flask import jsonify

# configure_logging("./app.log")
configure_logging(None)
api_configure_logging(None)

app = Flask(__name__)

config_path = "./config.ini"
config_args = parse_config_arguments(config_path)
print(config_args)

NEURDB_CONNECTOR = None
#
#     = NeurDBModelHandler(
#     {
#         "db_name": config_args.db_name,
#         "db_user": config_args.db_user,
#         "db_host": config_args.db_host,
#         "db_port": config_args.db_port,
#         # "password": config_args.db_password,
#     }
# )

# shared global contexts among tasks.
with app.app_context():
    app.config["config_args"] = config_args
    app.config["db_connector"] = NEURDB_CONNECTOR
    app.config["data_cache"] = ContextStates[DataCache]()
    app.config["dispatchers"] = ContextStates[LibSvmDataDispatcher]()
    app.config["clients"] = {}

app.before_request(before_request_func)
app.after_request(after_request_func)

# register http svc
app.register_blueprint(train_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(finetune_bp)


@app.route("/test", methods=["GET"])
def testing_app():
    print("Test router")
    return jsonify("finish testing")


@app.route("/force_disconnect/<sid>", methods=["POST"])
def force_disconnect(sid):
    """
    HTTP endpoint to forcefully disconnect a client.
    :param sid: Session ID of the client to disconnect.
    """
    print(f"Received request to forcefully disconnect client: {sid}")
    socketio.server.manager.disconnect(sid, "/")
    app.config["clients"].pop(sid, None)
    app.config["data_cache"].remove(sid)
    app.config["dispatchers"].remove(sid)
    return jsonify({"status": "disconnected"}), 200


# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = "threading"
# monkey patching is necessary because this application uses a background thread
# if async_mode == 'eventlet':
#     import eventlet
#     eventlet.monkey_patch()
# elif async_mode == 'gevent':
#     from gevent import monkey
#     monkey.patch_all()
socketio.init_app(app, async_mode=async_mode)
socketio.on_namespace(NRDataManager("/"))

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=app.config["config_args"].server_port,
        allow_unsafe_werkzeug=True,
    )
