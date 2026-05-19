# Only export model class so pickle can resolve neurdbrt.model.armnet.model.ARMNetModel.
# No builder, no register_model, no app/dataloader imports.
from .model import ARMNetModel

__all__ = ["ARMNetModel"]
