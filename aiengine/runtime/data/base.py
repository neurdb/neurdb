from typing import Annotated, Any, Dict
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


NonEmptyStr = Annotated[str, Field(min_length=1)]


class RuntimeDataModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls.model_validate(data)


class ArrowRuntimeModel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"


class ColumnStype(str, Enum):
    """Semantic type tag stored in column metadata."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"


class MetadataKey(str, Enum):
    """Well-known keys inside ColumnSchema.metadata."""

    STYPE = "stype"
    STATS = "stats"
    CARDINALITY = "cardinality"
    MEAN = "mean"
    STD = "std"
    STRATEGY = "strategy"
