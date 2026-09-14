# Local/project imports
from common.base_config import BaseConfig
from common.config_imdb import IMDBConfig
from common.config_stack import StackConfig


def get_config(name: str):
    if name == "imdb":
        return IMDBConfig()
    elif name == "stack":
        return StackConfig()
    else:
        raise ValueError(f"Unknown dataset: {name}")
