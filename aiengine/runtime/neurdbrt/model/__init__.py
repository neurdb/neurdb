from .armnet import *
from .base import *
from .mlp_clf import *


def build_model(model_name: str, config_args) -> BuilderBase:
    model = None
    if model_name == "armnet":
        model = ARMNetModelBuilder(config_args)
    # if model_name == "mlp":
    # model = MLPBuilder(config_args)
    return model
