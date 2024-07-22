import threading
from typing import Optional


class DataCache:
    def __init__(self, maxsize=20):
        self.lock = threading.Lock()
        self.cache = {
            'train': [],
            'evaluate': [],
            'test': [],
            'inference': []
        }
        self.maxsize = maxsize

    def set(self, key: str, value: dict) -> bool:
        with self.lock:
            if key in self.cache and len(self.cache[key]) < self.maxsize:
                self.cache[key].append(value)
                return True
            return False

    def get(self, key: str) -> Optional[dict]:
        with self.lock:
            if key in self.cache and len(self.cache[key]) > 0:
                return self.cache[key].pop(0)
            return None

    def is_full(self) -> Optional[str]:
        with self.lock:
            for key, values in self.cache.items():
                if len(values) < self.maxsize:
                    return key
            return None

    def is_empty(self):
        with self.lock:
            return all(len(values) == 0 for values in self.cache.values())
