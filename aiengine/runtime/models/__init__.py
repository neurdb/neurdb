from models.armnet.builder import ARMNetModelBuilder
from models.base.builder import BuilderBase
from models.mlp_clf.builder import MLPBuilder


def build_model(model_name: str, config_args) -> BuilderBase:
    model = None
    if model_name == "armnet":
        model = ARMNetModelBuilder(config_args)
    # if model_name == "mlp":
    # model = MLPBuilder(config_args)
    return model
