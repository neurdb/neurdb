import time
from client_socket import SocketClient, generate_dataset, generate_id_only_dataset
from client_stream import test_train_endpoint


def main():
    base_url = "http://127.0.0.1:8090"
    dataset = generate_id_only_dataset()

    # Initialize the SocketClient
    socket_client = SocketClient(base_url)
    socket_client.dataset = dataset  # Set the dataset before running the client

    # Run the SocketClient in a separate thread to handle the asynchronous nature
    import threading
    socket_thread = threading.Thread(target=socket_client.run)
    socket_thread.start()

    time.sleep(1)

    # Once the dataset initialization is done, send the train task
    _client_id = socket_client.sid
    _epoch = 1
    _batch_num = 2
    _batch_size = 2  # Example batch size
    _model_name = "armnet"  # Example model name
    _dataset_name = "frappe"

    test_train_endpoint(
        _batch_size,
        _model_name,
        _dataset_name,
        _client_id,
        _epoch,
        _batch_num,
        _batch_num,
        _batch_num,
    )

    socket_client.disconnect()




if __name__ == "__main__":
    main()
