from flask_socketio import emit
from flask import current_app
from app.websocket import socketio


@socketio.on('client_data')
def receive_db_data(data: str):
    """
    Receive data from the database UDFs.
    :param data:
    :return:
    """
    data_queue = current_app.config['data_queue']
    if data_queue.add(data):
        emit('response', {'message': 'Data received and added to queue!'})
    else:
        emit('response', {'message': 'Queue is full, data not added.'})
