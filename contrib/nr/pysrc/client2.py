import socketio

# Create a SocketIO client
sio = socketio.Client()

# Define the event handler for 'request_data'
@sio.on('request_data')
def on_request_data(data):
    print("Received on_request_data")
    key = data.get('key')
    print(f"Received request_data for key: {key}")
    # Handle the request data logic here
    dataset = 'sample_data'  # Replace with actual data fetching logic
    sio.emit('receive_db_data', {'dataset_name': key, 'dataset': dataset})

@sio.event
def connect():
    print("Connected to the server")

@sio.event
def disconnect():
    print("Disconnected from the server")

# Connect to the SocketIO server
sio.connect('http://localhost:5000')

# Keep the client running
sio.wait()




