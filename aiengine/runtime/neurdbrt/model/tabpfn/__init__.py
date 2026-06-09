from neurdbrt.model import register_model

from .builder import TabPFNModelBuilder
from .runner import run_tabular_task
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


def neurdb_on_start():
    register_model("tabpfn", TabPFNModelBuilder)
