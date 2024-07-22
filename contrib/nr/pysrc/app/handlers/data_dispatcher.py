import queue
import threading
import time


class LibSvmDataQueue:
    def __init__(self, socketio, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.socketio = socketio
        self.thread = threading.Thread(target=self._background_thread)
        self.thread.daemon = True
        self.thread.start()

    def preprocess(self, data):
        # Implement your preprocessing logic here
        # For example, let's just return the data as is
        return data

    def add(self, data):
        try:
            processed_data = self.preprocess(data)
            self.queue.put(processed_data, timeout=1)
            return True
        except queue.Full:
            return False

    def delete(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def _is_full(self):
        return self.queue.full()

    def _is_empty(self):
        return self.queue.empty()

    def _background_thread(self):
        print("[LibSvmDataQueue] thread started...")
        while True:
            if not self._is_full():
                # ask for data
                print("[LibSvmDataQueue] fetching data...")
                self.socketio.emit('request_data')
            time.sleep(1)  # Adjust the sleep time as needed
