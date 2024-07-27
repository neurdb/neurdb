from flask import current_app, request
from flask_socketio import Namespace, emit
from flask_socketio import SocketIO
from cache import DataCache, Bufferkey, LibSvmDataDispatcher

socketio = SocketIO(ping_timeout=30, ping_interval=5, logger=False, engineio_logger=False)


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
        print(current_app.config['clients'])
        emit('message', {'data': sid}, room=sid)

    def on_disconnect(self):
        """
        Handle client disconnection event.
        Remove the client session ID and associated data from the server.
        """
        sid = request.sid
        print(f"{sid} Client disconnected: ")
        current_app.config['clients'].pop(sid, None)
        current_app.config["data_cache"].remove(sid)
        current_app.config["dispatchers"].remove(sid)

    def on_dataset_init(self, data: dict):
        """
        Handle dataset initialization event.
        1. Create data cache for a specific dataset.
        2. Create dispatcher and start it.
        :param data: Dictionary containing dataset information.
        :return:
        """
        socket_id = request.sid
        dataset_name = data["dataset_name"]
        nfeat = data['nfeat']
        nfield = data['nfield']

        # 1. Create data cache if not exist
        data_cache = current_app.config["data_cache"]
        if not data_cache.contains(socket_id, dataset_name):
            _cache = DataCache(dataset_name)
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

        emit('response', {'message': 'Done'})

    def on_receive_db_data(self, data: dict):
        """
        Handle the event of receiving database data.
        Add the received data to the appropriate cache queue.
        :param data: Dictionary containing dataset information and the actual data.
        """
        socket_id = request.sid
        print(f"[socket]: {socket_id} receive_db_data...")
        dataset_name = data["dataset_name"]
        dataset = data["dataset"]

        # Check if dispatcher is launched for this dataset
        dispatchers = current_app.config["dispatchers"]
        if not dispatchers.contains(socket_id, dataset_name):
            emit("response", {
                "message": f"dispatchers is not initialized for dataset {dataset_name} and client {socket_id}, "
                           f"wait for train/inference/finetune request"})
            return

        dispatcher = dispatchers.get(socket_id, dataset_name)
        if dispatcher and dispatcher.add(dataset):
            emit('response', {'message': 'Data received and added to queue!'})
        else:
            emit('response', {'message': 'Queue is full, data not added.'})


def emit_request_data(client_id: str):
    """
    Emit request_data event to clients.
    :param client_id: The client ID to send the request to.
    :return:
    """
    print("[socket]: emit_request_data with key={key}...")
    socketio.emit('request_data', {}, to=client_id)
