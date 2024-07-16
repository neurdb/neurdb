import requests
import json

# Define the server address and port
SERVER_URL = "http://localhost:8090"


def test_train_endpoint(file_path, batch_size, model_name):
    url = f"{SERVER_URL}/train"
    with open(file_path, 'rb') as file:
        files = {'libsvm_file': file}
        data = {
            'batch_size': batch_size,
            'model_name': model_name
        }
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print(f"Response from {url}:")
            response_json = response.json()
            print(response_json)
            return response_json.get('model_id')
        else:
            print(f"Failed to get a valid response from {url}. Status code: {response.status_code}")
            return None


def test_inference_endpoint(file_path, model_name, model_id):
    url = f"{SERVER_URL}/inference"
    with open(file_path, 'rb') as file:
        files = {'libsvm_file': file}
        data = {
            'model_name': model_name,
            'model_id': model_id
        }
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print(f"Response from {url}:")
            print(response.json())
        else:
            print(f"Failed to get a valid response from {url}. Status code: {response.status_code}")
            print("Response content:")
            print(response.content)


if __name__ == "__main__":
    # Test sending the libsvm file to train endpoint
    _file_path = './dataset/frappe/test.libsvm'
    _batch_size = 32  # Example batch size
    _model_name = 'armnet'  # Example model name

    _model_id = 1
    _model_id = test_train_endpoint(_file_path, _batch_size, _model_name)

    test_inference_endpoint(_file_path, _model_name, int(_model_id))
