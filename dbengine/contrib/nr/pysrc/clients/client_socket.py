import json
import socketio
import random


class SocketClient:
    def __init__(self, base_url):
        self.sio = socketio.Client()
        self.base_url = base_url
        self.sid = ""
        self.dataset = ""  # Initialize self.dataset

        # Define event handlers
        self.sio.event(self.connect)
        self.sio.event(self.disconnect)
        self.sio.on("connection")(self.on_connection)
        self.sio.on("message")(self.on_message)
        self.sio.on("response")(self.on_response)
        self.sio.on("request_data")(self.on_request_data)

    def connect(self):
        print("Connected to the server")
        # Once connected, emit the dataset initialization event
        self.test_dataset_init("frappe", 5500, 10)

    def disconnect(self):
        print("Disconnected from the server")

    def on_connection(self, data):
        self.sid = data.get("sid")
        print(f"Received session ID from server: {self.sid}\n")

    def on_message(self, data):
        self.sid = data.get("data")
        print(f"Received session ID from server: {self.sid}\n")

    def on_response(self, data):
        print(f"Server response: {data}\n")

    def on_request_data(self, data):
        print("Received on_request_data")
        if self.dataset:
            self.sio.emit("batch_data", json.dumps({"dataset_name": "frappe", "dataset": self.dataset}))
        else:
            print("Dataset is not set!")

    def test_dataset_init(self, dataset_name, nfeat, nfield):
        profiling_data = {"dataset_name": dataset_name, "nfeat": nfeat, "nfield": nfield,
                          "nbatch": 1000, "cache_num": 80}
        self.sio.emit("dataset_init", json.dumps(profiling_data))

    def run(self):
        try:
            self.sio.connect(self.base_url)
            self.sio.wait()
        except Exception as e:
            print(f"An error occurred: {e}")


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


def generate_id_only_dataset(target_rows: int = 4096) -> str:
    base_strings = [
        "0 204  4798  5041  5046  5053  5055  5058  5060  5073  5183 ",
        "1 42  1572  5042  5047  5053  5055  5058  5060  5070  5150 ",
        "1 282  2552  5044  5052  5054  5055  5058  5060  5072  5244 ",
        "0 215  1402  5039  5051  5054  5055  5058  5063  5069  5149 ",
        "0 346  2423  5043  5051  5054  5055  5058  5063  5088  5149 ",
        "0 391  2081  5039  5050  5054  5055  5058  5060  5088  5268 ",
        "0 164  3515  5042  5052  5053  5055  5058  5062  5074  5149 ",
        "0 4  1177  5044  5049  5054  5057  5058  5060  5071  5152 "
    ]
    _dataset = "\n".join(random.choice(base_strings) for _ in range(target_rows))
    return _dataset


def main():
    base_url = "http://127.0.0.1:8090"
    dataset = generate_dataset()
    client = SocketClient(base_url)
    client.dataset = dataset  # Set the dataset before running the client
    # Connect and run the client
    client.run()


if __name__ == "__main__":
    main()
