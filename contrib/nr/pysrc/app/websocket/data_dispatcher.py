from flask_socketio import emit
from flask import current_app
from app.websocket import socketio
from cache.data_cache import DataCache


@socketio.on('dataset_profiling')
def dataset_profiling(data: str):
    """
    Create data cache for a specirc dataset
    :param data:
    :return:
    """
    data_cache = current_app.config['data_cache']
    dataset_name, nfeat, nfield = data.split(",")
    # create datacache
    _cache = DataCache()
    _cache.dataset_statistics = (nfeat, nfield)
    data_cache[dataset_name] = _cache

    emit('response', {'message': 'Done'})


@socketio.on('receive_db_data')
def receive_db_data(data: str):
    """
    Receive data from the database UDFs.
    :param data:
    :return:
    """
    dataset_name, dataset = data.split(",")
    _cache = current_app.config['data_cache'][dataset_name]

    if _cache.add(dataset):
        emit('response', {'message': 'Data received and added to queue!'})
    else:
        emit('response', {'message': 'Queue is full, data not added.'})
