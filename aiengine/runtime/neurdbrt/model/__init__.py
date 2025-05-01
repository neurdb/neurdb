from typing import Type

from ..log import logger
from .base import *

_name_builder_map = {}


def register_model(model_name: str, builder: Type[BuilderBase]):
    logger.info("registering model", model_name=model_name)
    _name_builder_map[model_name] = builder


def build_model(model_name: str, config_args) -> BuilderBase:
    logger.info("building model", model_name=model_name)

    builder = _name_builder_map.get(model_name)
    if not builder:
        raise ValueError(f"model {model_name} not found")

    return builder(config_args)
