from app.handlers.data_dispatcher import LibSvmDataDispatcher
from app.websocket.data_socket import socketio
from flask import current_app, g
from dataloader.steam_libsvm_dataset import StreamingDataSet


def before_request_func():
    print("before_request executing!")
    g.data_dispatcher = LibSvmDataDispatcher(socketio=socketio)


def after_request_func(response):
    print("after_request executing!")
    g.data_dispatcher.stop()
    g.data_dispatcher = None


def before_execute(dataset_name: str, data_key: str) -> bool:
    """
    Start LibSvmDataDispatcher and create StreamingDataSet
    :param dataset_name:
    :param data_key: train, infernece
    :return:
    """
    print("before_execute executing!")
    # get the data cache for that dataset
    data_cache = current_app.config['data_cache']
    if dataset_name not in data_cache:
        return False
    _cache = data_cache[dataset_name]

    # assign the data dispaccher
    g.data_dispatcher.set_dispatcher_target(_cache)
    if not g.data_dispatcher.start():
        return False

    # register it as global variables
    # todo: multiple request share same dispatcher, add refercne count
    dispatchers = current_app.config['dispatchers']
    if dataset_name not in dispatchers:
        dispatchers[dataset_name] = dispatchers

    # create dataset
    data = StreamingDataSet(_cache, data_key=data_key)
    g.data_loader = data
    return True
