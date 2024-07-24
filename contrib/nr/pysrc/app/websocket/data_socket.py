from flask import current_app, request
from flask_socketio import Namespace, emit
from flask_socketio import SocketIO
from cache import DataCache, Bufferkey, LibSvmDataDispatcher

socketio = SocketIO(ping_timeout=30, ping_interval=5, logger=False, engineio_logger=False)


class NRDataManager(Namespace):

    def on_connect(self):
        sid = request.sid
        current_app.config['clients'][sid] = sid

        print(f"Client connected: {sid}")
        print(current_app.config['clients'])
        emit('message', {'data': sid}, room=sid)

    def on_disconnect(self):
        sid = request.sid
        print(f"{sid} Client disconnected: ")
        current_app.config["clients"].pop(sid)
        current_app.config["data_cache"].pop(sid)
        current_app.config["dispatchers"].pop(sid)

    def on_dataset_init(self, data: dict):
        """
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
        if socket_id not in current_app.config["data_cache"]:
            current_app.config["data_cache"][socket_id] = {}

        if dataset_name not in current_app.config["data_cache"]:
            _cache = DataCache(dataset_name)
            _cache.dataset_statistics = (nfeat, nfield)
            current_app.config["data_cache"][socket_id][dataset_name] = _cache
        else:
            _cache = current_app.config["data_cache"][socket_id][dataset_name]

        # 2. Create dispatcher
        if socket_id not in current_app.config["dispatchers"]:
            current_app.config["dispatchers"][socket_id] = {}

        # Check if the dataset exists for the client in the dispatchers dictionary
        if dataset_name not in current_app.config["dispatchers"][socket_id]:
            _data_dispatcher = LibSvmDataDispatcher()
            current_app.config["dispatchers"][socket_id][dataset_name] = _data_dispatcher
            _data_dispatcher.bound_client_to_cache(_cache, socket_id)
            _data_dispatcher.start(emit_request_data)

        emit('response', {'message': 'Done'})

    def on_receive_db_data(self, data: dict):
        """
        Receive data from the database UDFs.
        :param data: Dictionary containing dataset name and data.
        :return:
        """
        socket_id = request.sid
        print(f"[socket]: {socket_id} receive_db_data...")
        dataset_name = data["dataset_name"]
        ml_stage = data["ml_stage"]
        dataset = data["dataset"]

        # check the ml_stage can be reconginzed
        ml_stage = Bufferkey.get_key_by_value(ml_stage)
        if not ml_stage:
            emit("response", {
                "message": f"{ml_stage} cannot be recognized, "
                           f"only support 'train', 'evaluate', 'test', 'inference'"})
            return

        # check dispatcher is launched for this datasets
        if socket_id not in current_app.config["dispatchers"]:
            emit("response", {
                "message": f"dispatchers is not initialized for dataset {dataset_name} and client {socket_id}, "
                           f"wait for train/infernce/finetune request"})
            return

        dispatcher = current_app.config['dispatchers'][socket_id][dataset_name]

        if dispatcher.add(ml_stage, dataset):
            emit('response', {'message': 'Data received and added to queue!'})
        else:
            emit('response', {'message': 'Queue is full, data not added.'})


def emit_request_data(key: Bufferkey, client_id: str):
    """
    Emit request_data event to clients
    :param key: Bufferkey.TRAIN_KEY etc
    :param client_id:
    :return:
    """
    print(f"[socket]: emit_request_data with key={key}...")
    socketio.emit('request_data', {'key': key.value}, to=client_id)
