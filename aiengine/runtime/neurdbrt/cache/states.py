import threading
from typing import Dict, Generic, Optional, TypeVar, Union

T = TypeVar("T")  # Generic type


class ContextStates(Generic[T]):
    """
    ContextStates define global states and allows thread safe access
    """

    def __init__(self):
        self.state: Dict[str, Dict[str, T]] = {}
        self.lock = threading.Lock()

    def add(self, outer_key: str, inner_key: str, value: T):
        """
        Add a value to the two-level key dict
        :param outer_key:
        :param inner_key:
        :param value:
        :return:
        """
        with self.lock:
            if outer_key not in self.state:
                self.state[outer_key] = {}
            self.state[outer_key][inner_key] = value
            print(f"Added ({outer_key}, {inner_key}) = {value}")

    def remove(self, outer_key: str):
        """
        Remove all entries under the given outer key
        :param outer_key:
        :return:
        """
        with self.lock:
            if outer_key in self.state:
                del self.state[outer_key]
                print(f"Removed all contents under outer key ({outer_key})")
            else:
                print(f"Error: Outer key ({outer_key}) not found")

    def get(
        self, outer_key: str, inner_key: Optional[str] = None
    ) -> Union[Optional[T], Dict[str, T]]:
        """
        Get the inner dictionary or value for a given outer key
        :param outer_key:
        :param inner_key:
        :return:
        """
        with self.lock:
            if outer_key not in self.state:
                print(f"Outer key ({outer_key}) not found")
                return None

            if inner_key is None:
                return self.state[outer_key]

            if inner_key not in self.state[outer_key]:
                print(
                    f"Inner key ({inner_key}) not found under outer key ({outer_key})"
                )
                return None

            return self.state[outer_key][inner_key]

    def contains(self, outer_key: str, inner_key: str) -> bool:
        """
        Check if both the given outer and inner key exist in the state
        :param outer_key:
        :param inner_key:
        :return:
        """
        with self.lock:
            if outer_key not in self.state:
                return False

            inner_state = self.state[outer_key]
            if inner_key not in inner_state:
                return False

            return True
