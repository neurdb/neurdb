from neurdbrt.model import register_model

from .builder import *
from .model import *


def neurdb_on_start():
    register_model("armnet", ARMNetModelBuilder)
