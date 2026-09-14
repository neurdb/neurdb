from neurdbrt.model import register_model

from . import session_store
from .builder import TabPFNModelBuilder
from .model import BINARY, REGRESSION, TabPFNPredictor
from .preprocess import (
    CATEGORICAL,
    DROP,
    NUMERICAL,
    TEXT,
    TIMESTAMP,
    PreprocessResult,
    TabularPreprocessor,
    infer_column_stypes,
)
from .runner import run_tabular_task
from .stateful import StatefulTabPFN, infer_task_type, pg_types_to_hints


def neurdb_on_start():
    register_model("tabpfn", TabPFNModelBuilder)
