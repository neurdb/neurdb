from flask import current_app, request
from cache.data_cache import DataCache
from flask_socketio import Namespace, emit
from flask_socketio import SocketIO

socketio = SocketIO(ping_timeout=30, ping_interval=5)


class NRDataManager(Namespace):

    def on_connect(self):
        sid = request.sid
        current_app.config['clients'][sid] = request.namespace

        print(f"Client connected: {sid}")
        emit('message', {'data': sid}, room=sid)

    def on_disconnect(self):
        sid = request.sid
        print(f"{sid} Client disconnected: ")
        current_app.config['clients'].pop(sid)

    def on_dataset_profiling(self, data: dict):
        """
        Create data cache for a specific dataset.
        :param data: Dictionary containing dataset information.
        :return:
        """
        data_cache = current_app.config['data_cache']
        dataset_name = data['dataset_name']
        nfeat = data['nfeat']
        nfield = data['nfield']

        # Create data cache
        _cache = DataCache()
        _cache.dataset_statistics = (nfeat, nfield)
        data_cache[dataset_name] = _cache

        emit('response', {'message': 'Done'})

    def on_receive_db_data(self, data: dict):
        """
        Receive data from the database UDFs.
        :param data: Dictionary containing dataset name and data.
        :return:
        """
        print("[socket]: receive_db_data...")
        dataset_name = data['dataset_name']
        dataset = data['dataset']
        if dataset_name not in current_app.config['dispatchers']:
            emit('response', {'message': 'dispatchers is not initialized.'})
            return

        dispatcher = current_app.config['dispatchers'][dataset_name]

        if dispatcher.set(dataset):
            emit('response', {'message': 'Data received and added to queue!'})
        else:
            emit('response', {'message': 'Queue is full, data not added.'})


def emit_request_data(key: str, sid: str):
    """
    Emit request_data event to a specific client
    :param key: The key to be sent
    :param sid: The session ID of the specific client
    :return: None
    """
    client = current_app.config['clients'][sid]
    print(f"[socket]: emit_request_data to client {sid} with key {key}...")
    client.emit(f"request_data', {'key': {key}}")
