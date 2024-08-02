import requests

# Define the server address and port
SERVER_URL = "http://localhost:8090"


def test_train_endpoint(
        batch_size,
        model_name,
        dataset_name,
        client_id,
        epoch,
        train_batch_num,
        eva_batch_num,
        test_batch_num,
):
    url = f"{SERVER_URL}/train"
    data = {
        "batch_size": batch_size,
        "model_name": model_name,
        "table_name": dataset_name,
        "client_socket_id": client_id,
        "epoch": epoch,
        "train_batch_num": train_batch_num,
        "eva_batch_num": eva_batch_num,
        "test_batch_num": test_batch_num,
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print(f"Response from {url}:")
        response_json = response.json()
        print(response_json)
        return response_json.get("model_id")
    else:
        print(
            f"Failed to get a valid response from {url}. Status code: {response.status_code}"
        )
        return None


def test_inference_endpoint(model_name, model_id, client_id, inf_batch_num):
    url = f"{SERVER_URL}/inference"
    data = {
        "model_name": model_name,
        "model_id": model_id,
        "client_socket_id": client_id,
        "inf_batch_num": inf_batch_num,
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print(f"Response from {url}:")
        print(response.json())
    else:
        print(
            f"Failed to get a valid response from {url}. Status code: {response.status_code}"
        )
        print("Response content:")
        print(response.content)


if __name__ == "__main__":
    _client_id = "f-QqXfX--8I0GsbBAAAB"
    # Test sending the libsvm data to train endpoint
    _epoch = 1
    _batch_num = 3
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

    test_inference_endpoint(_model_name, 0, _client_id, _batch_num)

    "curl -X POST http://127.0.0.1:5000/force_disconnect/<sid>"
