import numpy as np
from multiprocessing import shared_memory
import torch.nn as nn
import torch
import tempfile
import os
import io


def get_data_from_shared_memory_int(n_rows):
    shm = shared_memory.SharedMemory(name="my_shared_memory")
    data = np.frombuffer(shm.buf, dtype=np.float32)
    data = data.reshape(n_rows, -1)
    return data


def save_model_weight(model: nn.Module, model_repo: str) -> (bytes, str):
    model_path = os.path.join(model_repo, "mlp_model.pt")
    torch.save(model.state_dict(), model_path)

    with open(model_path, 'rb') as file:
        model_binary = file.read()

    return model_binary, model_path


def load_model_weight(model_binary: bytes, mode_ins: nn.Module):
    buffer = io.BytesIO(model_binary)
    mode_ins.load_state_dict(torch.load(buffer))
    mode_ins.eval()  # Set the model to evaluation mode


def save_model_jit(model: nn.Module, model_repo: str) -> (str, str, str):
    model_path = model_repo + "/mlp_model.pt"
    torch.jit.save(torch.jit.script(model), model_path)

    with open(model_path, 'rb') as file:
        model_binary = file.read()

    return model_binary, model_path


def load_model_jit(model_binary: bytes) -> torch.nn.Module:
    buffer = io.BytesIO(model_binary)
    model = torch.jit.load(buffer)
    return model
