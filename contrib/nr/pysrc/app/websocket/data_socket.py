from flask import current_app, request
from cache.data_cache import DataCache
from flask_socketio import Namespace, emit
from flask_socketio import SocketIO
from cache.data_cache import Bufferkey

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
        current_app.config['clients'].pop(sid)

    def on_dataset_init(self, data: dict):
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
        _cache = DataCache(dataset_name)
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
        if dataset_name not in current_app.config["dispatchers"]:
            emit("response", {
                "message": f"dispatchers is not initialized for dataset {dataset_name}, "
                           f"wait for train/infernce/finetune request"})
            return

        dispatcher = current_app.config['dispatchers'][dataset_name]

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
