import threading
from typing import Dict, TypeVar, Generic

T = TypeVar('T')  # Generic type


class ContextStates(Generic[T]):
    def __init__(self):
        self.data_cache: Dict[str, Dict[str, T]] = {}
        self.lock = threading.Lock()

    def add(self, outer_key: str, inner_key: str, value: T):
        with self.lock:
            if outer_key not in self.data_cache:
                self.data_cache[outer_key] = {}
            self.data_cache[outer_key][inner_key] = value
            print(f"Added ({outer_key}, {inner_key}) = {value}")

    def remove(self, outer_key: str, inner_key: str):
        with self.lock:
            if outer_key in self.data_cache and inner_key in self.data_cache[outer_key]:
                del self.data_cache[outer_key][inner_key]
                # Remove outer_key if empty
                if not self.data_cache[outer_key]:
                    del self.data_cache[outer_key]
                print(f"Removed ({outer_key}, {inner_key})")
            else:
                print(f"Key ({outer_key}, {inner_key}) not found")

    def get(self, outer_key: str, inner_key: str) -> T:
        with self.lock:
            return self.data_cache.get(outer_key, {}).get(inner_key)

    def __contains__(self, outer_key: str) -> bool:
        with self.lock:
            return outer_key in self.data_cache

    def __getitem__(self, outer_key: str) -> Dict[str, T]:
        with self.lock:
            return self.data_cache[outer_key]
