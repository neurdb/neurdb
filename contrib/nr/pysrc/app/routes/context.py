from flask import current_app, g
from dataloader.steam_libsvm_dataset import StreamingDataSet
from cache.data_cache import Bufferkey


def before_request_func():
    print("before_request executing!")


def after_request_func(response):
    print("after_request executing!")
    g.data_dispatcher.stop()
    g.data_dispatcher = None


def before_execute(dataset_name: str, data_key: Bufferkey, client_id: str) -> (bool, str):
    """
    1. check socket client is connected
    2. check data cache exist for dataset_name
    3. create steaming data loader
    :param dataset_name:
    :param data_key: train, infernece
    :param client_id: socket client id
    :return:
    """
    print("before_execute executing!")

    # 1. check the client is connected
    if client_id not in current_app.config['clients']:
        return False, f"client {client_id} is not registered by socket, no data here !"

    # 2. check the data cache for that dataset
    data_cache = current_app.config['data_cache']
    if dataset_name not in data_cache:
        return False, f"Dataset {dataset_name} is not registred in web-socket"
    _cache = data_cache[dataset_name]

    # 3. create dataset
    g.data_loader = StreamingDataSet(_cache, data_key=data_key)
    return True, ""
