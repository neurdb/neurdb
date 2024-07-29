import ast

from flask import current_app, request
from flask_socketio import SocketIO, Namespace, emit, disconnect
from cache import DataCache, LibSvmDataDispatcher

socketio = SocketIO(
    ping_timeout=30, ping_interval=5, logger=False, engineio_logger=False
)


class NRDataManager(Namespace):
    """
    NRDataManager register some socket endpoints
    """

    def on_connect(self):
        """
        Handle client connection event.
        Store the client session ID and notify the client.
        """
        sid = request.sid
        current_app.config["clients"][sid] = sid

        print(f"Client connected: {sid}")
        _current_clients = current_app.config["clients"]
        print(f"Current registered clients: {_current_clients}")
        emit("connection", {"sid": sid}, room=sid)

    # todo: this cannot connected by c client.
    def on_disconnect(self):
        """
        Handle client disconnection event.
        Remove the client session ID and associated data from the server.
        """
        try:
            sid = request.sid
            print(f"{sid} Client disconnected: ")
            current_app.config["clients"].pop(sid, None)
            current_app.config["data_cache"].remove(sid)
            current_app.config["dispatchers"].remove(sid)
        except Exception as e:
            print(f"Error {e}")

    def on_dataset_init(self, data: str):
        """
        Handle dataset initialization event.
        1. Create data cache for a specific dataset.
        2. Create dispatcher and start it.
        :param data: Dictionary containing dataset information.
        :return:
        """
        # str to dict
        data = ast.literal_eval(data)

        socket_id = request.sid
        dataset_name = data["dataset_name"]
        nfeat = data["nfeat"]
        nfield = data["nfield"]
        total_batch_num = data["nbatch"]
        cache_num = data["cache_num"]

        # 1. Create data cache if not exist
        data_cache = current_app.config["data_cache"]
        if not data_cache.contains(socket_id, dataset_name):
            _cache = DataCache(
                dataset_name=dataset_name,
                total_batch_num=total_batch_num,
                maxsize=cache_num,
            )
            _cache.dataset_statistics = (nfeat, nfield)
            data_cache.add(socket_id, dataset_name, _cache)
        else:
            _cache = data_cache.get(socket_id, dataset_name)

        # 2. Create dispatcher if not exist
        dispatchers = current_app.config["dispatchers"]
        if not dispatchers.contains(socket_id, dataset_name):
            _data_dispatcher = LibSvmDataDispatcher()
            dispatchers.add(socket_id, dataset_name, _data_dispatcher)
            _data_dispatcher.bound_client_to_cache(_cache, socket_id)
            _data_dispatcher.start(emit_request_data)

        emit("dataset_init", {"message": "Done"})

    def on_batch_data(self, data: str):
        """
        Handle the event of receiving database data.
        Add the received data to the appropriate cache queue.
        :param data: Dictionary containing dataset information and the actual data.
        """
        socket_id = request.sid
        print(f"[socket]: {socket_id} receive_db_data...")
        data = ast.literal_eval(data)

        dataset_name = data["dataset_name"]
        dataset = data["dataset"]

        # Check if dispatcher is launched for this dataset
        dispatchers = current_app.config["dispatchers"]
        if not dispatchers.contains(socket_id, dataset_name):
            print(f"dispatchers is not initialized for dataset {dataset_name} and client {socket_id}, "
                  f"wait for train/inference/finetune request")
            emit(
                "response",
                {
                    "message": f"dispatchers is not initialized for dataset {dataset_name} and client {socket_id}, "
                               f"wait for train/inference/finetune request"
                },
            )
        else:
            dispatcher = dispatchers.get(socket_id, dataset_name)
            if dispatcher and dispatcher.add(dataset):
                print("Data received and added to queue!")
                emit("response", {"message": "Data received and added to queue!"})
            else:
                emit("response", {"message": "Queue is full, data not added."})

    def force_disconnect(self):
        """
        Forcefully disconnect a client.
        :param sid: Session ID of the client to disconnect.
        """
        sid = request.sid
        print(f"Forcefully disconnecting client: {sid}")
        disconnect(sid)


def emit_request_data(client_id: str):
    """
    Emit request_data event to clients.
    :param client_id: The client ID to send the request to.
    :return:
    """

    socketio.emit("request_data", {}, to=client_id)
