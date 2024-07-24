import threading
from typing import Dict, TypeVar, Generic, Optional, Union
from functools import singledispatchmethod

T = TypeVar('T')  # Generic type


class ContextStates(Generic[T]):
    def __init__(self):
        self.state: Dict[str, Dict[str, T]] = {}
        self.lock = threading.Lock()

    def add(self, outer_key: str, inner_key: str, value: T):
        with self.lock:
            if outer_key not in self.state:
                self.state[outer_key] = {}
            self.state[outer_key][inner_key] = value
            print(f"Added ({outer_key}, {inner_key}) = {value}")

    def remove(self, outer_key: str):
        with self.lock:
            if outer_key in self.state:
                del self.state[outer_key]
                print(f"Removed all contents under outer key ({outer_key})")
            else:
                print(f"Outer key ({outer_key}) not found")

    @singledispatchmethod
    def get(self, outer_key: str) -> Union[Optional[T], Dict[str, T]]:
        with self.lock:
            if outer_key not in self.state:
                print(f"Outer key ({outer_key}) not found")
                return None
            return self.state[outer_key]

    @get.register
    def _(self, outer_key: str, inner_key: str) -> Optional[T]:
        with self.lock:
            if outer_key not in self.state:
                print(f"Outer key ({outer_key}) not found")
                return None

            if inner_key not in self.state[outer_key]:
                print(f"Inner key ({inner_key}) not found under outer key ({outer_key})")
                return None

            return self.state[outer_key][inner_key]

    def contains(self, outer_key: str, inner_key: str) -> bool:
        with self.lock:
            if outer_key not in self.state:
                return False

            inner_state = self.state[outer_key]
            if inner_key not in inner_state:
                return False

            return True
