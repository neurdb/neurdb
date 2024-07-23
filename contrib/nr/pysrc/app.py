import time
from connection import NeurDBModelHandler
from logger.logger import configure_logging
from flask import Flask, g
from shared_config.config import parse_config_arguments
from app.routes import train_bp, inference_bp, finetune_bp
from app.websocket.data_socket import NRDataManager
from app.routes.context import before_request_func, after_request_func
from app.websocket.data_socket import socketio

app = Flask(__name__)

# Load config and initialize once
config_path = "./config.ini"
config_args = parse_config_arguments(config_path)
configure_logging("./logs/app.log")

# NEURDB_CONNECTOR = NeurDBModelHandler(
#     {
#         "db_name": config_args.db_name,
#         "db_user": config_args.db_user,
#         "db_host": config_args.db_host,
#         "db_port": config_args.db_port,
#         # "password": config_args.db_password,
#     }
# )
NEURDB_CONNECTOR = None

# shared global contexts among tasks.
data_cache = {}
dispatchers = {}
clients = {}

with app.app_context():
    app.config['config_args'] = config_args
    app.config['db_connector'] = NEURDB_CONNECTOR
    app.config['data_cache'] = data_cache
    app.config['dispatchers'] = dispatchers
    app.config['clients'] = clients

app.before_request(before_request_func)
app.after_request(after_request_func)

# register http svc
app.register_blueprint(train_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(finetune_bp)

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None
socketio.init_app(app, async_mode=async_mode)
socketio.on_namespace(NRDataManager('/'))

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=app.config['config_args'].server_port)
