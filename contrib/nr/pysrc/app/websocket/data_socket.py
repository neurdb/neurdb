from flask_socketio import emit
from flask import current_app, request
from app.websocket import socketio
from cache.data_cache import DataCache


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    socketio.emit('message', {'data': sid}, room=sid)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"{sid} Client disconnected: ")


@socketio.on('dataset_profiling')
def dataset_profiling(data: dict):
    """
    Create data cache for a specific dataset.
    :param data: Dictionary containing dataset information.
    :return:
    """
    data_cache = current_app.config['data_cache']
    dataset_name = data['dataset_name']
    nfeat = data['nfeat']
    nfield = data['nfield']

    # create datacache
    _cache = DataCache()
    _cache.dataset_statistics = (nfeat, nfield)
    data_cache[dataset_name] = _cache

    emit('response', {'message': 'Done'})


@socketio.on('receive_db_data')
def receive_db_data(data: dict):
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
    print(f"[socket]: emit_request_data to client {sid} with key {key}...")
    socketio.emit('request_data', {'key': key}, room=sid)
