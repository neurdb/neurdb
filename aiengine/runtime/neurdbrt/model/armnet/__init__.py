from .builder import *
from .model import *

from neurdbrt.model import register_model

def neurdb_on_start():
    register_model("armnet", ARMNetModelBuilder)
