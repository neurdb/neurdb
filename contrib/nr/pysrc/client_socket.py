import json
import time
import socketio

# Global dataset string
dataset = """0 204:1 4798:1 5041:1 5046:1 5053:1 5055:1 5058:1 5060:1 5073:1 5183:1\n
    1 42:1 1572:1 5042:1 5047:1 5053:1 5055:1 5058:1 5060:1 5070:1 5150:1\n
    1 282:1 2552:1 5044:1 5052:1 5054:1 5055:1 5058:1 5060:1 5072:1 5244:1\n
    0 215:1 1402:1 5039:1 5051:1 5054:1 5055:1 5058:1 5063:1 5069:1 5149:1\n
    0 346:1 2423:1 5043:1 5051:1 5054:1 5055:1 5058:1 5063:1 5088:1 5149:1\n
    0 391:1 2081:1 5039:1 5050:1 5054:1 5055:1 5058:1 5060:1 5088:1 5268:1\n
    0 164:1 3515:1 5042:1 5052:1 5053:1 5055:1 5058:1 5062:1 5074:1 5149:1\n
    0 4:1 1177:1 5044:1 5049:1 5054:1 5057:1 5058:1 5060:1 5071:1 5152:1"""

# Socket.IO client
sio = socketio.Client()


# Define the event handler for the 'message' event to receive the session ID
@sio.on('message')
def on_message(data):
    sid = data.get('data')
    print(f"Received session ID from server: {sid}\n")


# Define the event handler for the 'response' event
@sio.on('response')
def on_response(data):
    print(f"Server response: {data}\n")


# Define the event handler for the 'request_data' event
@sio.on('request_data')
def on_request_data(data):
    print("Received on_request_data")
    # Handle the request data logic here
    sio.emit('batch_data',
             json.dumps({"dataset_name": "frappe",
                         "dataset": dataset}))


@sio.event
def connect():
    print("Connected to the server")


@sio.event
def disconnect():
    print("Disconnected from the server")


def test_dataset_init(dataset_name, nfeat, nfield):
    profiling_data = {
        'dataset_name': dataset_name,
        'nfeat': nfeat,
        'nfield': nfield
    }
    sio.emit('dataset_init', json.dumps(profiling_data))


def test_receive_db_data(dataset_name, dataset):
    db_data = {
        'dataset_name': dataset_name,
        'dataset': dataset
    }
    sio.emit('batch_data', json.dumps(db_data))


# Connect to the Socket.IO server
sio.connect("http://127.0.0.1:8090")

# test_dataset_init('frappe', 5500, 10)
# test_receive_db_data('frappe', dataset)
time.sleep(2)

# Disconnect after sending the test data
sio.disconnect()

# Keep the client running
sio.wait()
