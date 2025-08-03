import ctypes
import gc
import json
import os
from typing import Any

import numpy as np
import torch

from . import env

__all__ = ["abspath", "write_json", "read_json"]


def abspath(*paths: str):
    import auto_pipeline
    return os.path.abspath(os.path.join(auto_pipeline.__path__[0], *paths))


def write_json(path: str, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f)


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


_libc = ctypes.CDLL("libc.so.6")


def clean_mem():
    del gc.garbage[:]
    gc.collect()
    _libc.malloc_trim(0)


def device_tensor(data: Any, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if not isinstance(data, torch.Tensor) and not isinstance(data, np.ndarray):
        data = np.array(data)

    return torch.tensor(data, dtype=dtype).to(env.DEVICE)
