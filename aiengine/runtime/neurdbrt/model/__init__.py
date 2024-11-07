from .armnet import *
from .base import *
from .mlp_clf import *
from ..log import logger

def build_model(model_name: str, config_args) -> BuilderBase:
    logger.info("building model", model_name=model_name)
    
    model = None
    if model_name == "armnet":
        model = ARMNetModelBuilder(config_args)
    # if model_name == "mlp":
    # model = MLPBuilder(config_args)
    return model
