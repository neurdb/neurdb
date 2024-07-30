import json
import socketio


def generate_dataset(target_rows: int = 4096) -> str:
    base_string = """0 204:1 4798:1 5041:1 5046:1 5053:1 5055:1 5058:1 5060:1 5073:1 5183\n
                    1 42:1 1572:1 5042:1 5047:1 5053:1 5055:1 5058:1 5060:1 5070:1 5150:1"""

    rows_per_string = base_string.count('\n') + 1
    repeat_count = (target_rows // rows_per_string) + 1
    final_string = base_string * repeat_count
    lines = final_string.strip().split('\n')
    final_string = '\n'.join(lines[:target_rows])
    print(f"Length of final string: {len(final_string)} characters")
    return final_string


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


base_url = "http://127.0.0.1:8090"
# Connect to the Socket.IO server
sio.connect(base_url)

# Disconnect after sending the test data
# sio.disconnect()

# Keep the client running
sio.wait()
