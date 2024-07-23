from flask_socketio import SocketIO

socketio = SocketIO(ping_timeout=30, ping_interval=5)
