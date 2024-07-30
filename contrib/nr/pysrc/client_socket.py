import json
import socketio
import random


def generate_dataset(target_rows: int = 4096) -> str:
    base_strings = [
        "0 204:1 4798:1 5041:1 5046:1 5053:1 5055:1 5058:1 5060:1 5073:1 5183:1",
        "1 42:1 1572:1 5042:1 5047:1 5053:1 5055:1 5058:1 5060:1 5070:1 5150:1",
        "1 282:1 2552:1 5044:1 5052:1 5054:1 5055:1 5058:1 5060:1 5072:1 5244:1",
        "0 215:1 1402:1 5039:1 5051:1 5054:1 5055:1 5058:1 5063:1 5069:1 5149:1",
        "0 346:1 2423:1 5043:1 5051:1 5054:1 5055:1 5058:1 5063:1 5088:1 5149:1",
        "0 391:1 2081:1 5039:1 5050:1 5054:1 5055:1 5058:1 5060:1 5088:1 5268:1",
        "0 164:1 3515:1 5042:1 5052:1 5053:1 5055:1 5058:1 5062:1 5074:1 5149:1",
        "0 4:1 1177:1 5044:1 5049:1 5054:1 5057:1 5058:1 5060:1 5071:1 5152:1"
    ]
    _dataset = "\n".join(random.choice(base_strings) for _ in range(target_rows))
    return _dataset


dataset = generate_dataset()
# Socket.IO client
sio = socketio.Client()
sid = ""


@sio.event
def connect():
    print("Connected to the server")


@sio.event
def disconnect():
    print("Disconnected from the server")


# Define the event handler for the 'message' event to receive the session ID
@sio.on("connection")
def on_connection(data):
    global sid
    sid = data.get("sid")
    print(f"Received session ID from server: {sid}\n")


# Define the event handler for the 'message' event to receive the session ID
@sio.on("message")
def on_message(data):
    global sid
    sid = data.get("data")
    print(f"Received session ID from server: {sid}\n")


# Define the event handler for the 'response' event
@sio.on("response")
def on_response(data):
    print(f"Server response: {data}\n")


# Define the event handler for the 'request_data' event
@sio.on("request_data")
def on_request_data(data):
    print("Received on_request_data")
    # Handle the request data logic here
    sio.emit("batch_data", json.dumps({"dataset_name": "frappe", "dataset": dataset}))


def test_dataset_init(dataset_name, nfeat, nfield):
    profiling_data = {"dataset_name": dataset_name, "nfeat": nfeat, "nfield": nfield,
                      "nbatch": 1000, "cache_num": 80}
    sio.emit("dataset_init", json.dumps(profiling_data))


base_url = "http://127.0.0.1:8090"
# Connect to the Socket.IO server
sio.connect(base_url)

# Disconnect after sending the test data
# sio.disconnect()

test_dataset_init("frappe", 5500, 10)

# Keep the client running
sio.wait()
