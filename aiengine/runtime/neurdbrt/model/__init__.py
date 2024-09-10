from .armnet import *
from .base import *
from .dnn import *


def build_model(model_name: str, config_args) -> BuilderBase:
    _model = None
    if model_name == "armnet":
        _model = ARMNetModelBuilder(config_args)
    if model_name == "dnn":
        _model = DNNModelBuilder(config_args)
    return _model
