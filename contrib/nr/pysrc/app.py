from connection import NeurDBModelHandler
from logger.logger import configure_logging
from flask import Flask, g
from shared_config.config import parse_config_arguments
from app.routes import train_bp, inference_bp, finetune_bp
from app.handlers.data_dispatcher import LibSvmDataQueue
from app.websocket import socketio

app = Flask(__name__)

# Load config and initialize once
config_path = "./config/config.ini"
config_args = parse_config_arguments(config_path)
configure_logging("./logs/app.log")

NEURDB_CONNECTOR = NeurDBModelHandler(
    {
        "db_name": config_args.db_name,
        "db_user": config_args.db_user,
        "db_host": config_args.db_host,
        "db_port": config_args.db_port,
        # "password": config_args.db_password,
    }
)

data_queue = LibSvmDataQueue(socketio, maxsize=100)

# define global contexts
with app.app_context():
    app.config['config_args'] = config_args
    app.config['db_connector'] = NEURDB_CONNECTOR
    app.config['data_queue'] = data_queue

# register http svc
app.register_blueprint(train_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(finetune_bp)

# register socket svcs
socketio.init_app(app)

if __name__ == "__main__":
    # support WebSocket while preserving all standard HTTP functionalities.
    socketio.run(app, host="0.0.0.0", port=app.config['config_args'].server_port)
