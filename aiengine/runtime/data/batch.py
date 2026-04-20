from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import Field, field_validator, model_validator

from .base import RuntimeDataModel, NonEmptyStr

class BatchRole(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    TEST = "test"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECOMMENDATION = "recommendation"
    LINK_PREDICTION = "link_prediction"  # NOT USED CURRENTLY
    UNKNOWN = "unknown"


class TargetSpec(RuntimeDataModel):
    table: str
    column: str
    task_type: TaskType = TaskType.UNKNOWN
    timestamp_column: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("table")
    @classmethod
    def _validate_table(cls, table: str) -> str:
        if not table:
            raise ValueError("target table must not be empty")
        return table

    @field_validator("column")
    @classmethod
    def _validate_column(cls, column: str) -> str:
        if not column:
            raise ValueError("target column must not be empty")
        return column


class TableBatch(RuntimeDataModel):
    table: str
    columns: List[str]
    rows: Sequence[Sequence[Any]]
    row_ids: Optional[Sequence[Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("table")
    @classmethod
    def _validate_table(cls, table: str) -> str:
        if not table:
            raise ValueError("table batch table name must not be empty")
        return table

    @model_validator(mode="after")
    def _validate_shape(self) -> "TableBatch":
        if not self.columns:
            raise ValueError(f"table batch {self.table} must contain columns")
        column_count = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != column_count:
                raise ValueError(
                    f"row {index} in table batch {self.table} has {len(row)} "
                    f"values, expected {column_count}"
                )

        if self.row_ids is not None and len(self.row_ids) != len(self.rows):
            raise ValueError(
                f"table batch {self.table} row_ids length must match rows length"
            )
        return self

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


class TargetBatch(RuntimeDataModel):
    table: str
    column: str
    values: Sequence[Any]
    row_ids: Optional[Sequence[Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("table")
    @classmethod
    def _validate_table(cls, table: str) -> str:
        if not table:
            raise ValueError("target batch table name must not be empty")
        return table

    @field_validator("column")
    @classmethod
    def _validate_column(cls, column: str) -> str:
        if not column:
            raise ValueError("target batch column must not be empty")
        return column

    @model_validator(mode="after")
    def _validate_shape(self) -> "TargetBatch":
        if self.row_ids is not None and len(self.row_ids) != len(self.values):
            raise ValueError("target batch row_ids length must match values length")
        return self

    @property
    def row_count(self) -> int:
        return len(self.values)


class DataBatch(RuntimeDataModel):
    tables: Mapping[str, TableBatch]
    target: Optional[TargetBatch] = None
    role: BatchRole = BatchRole.UNKNOWN
    batch_id: Optional[int] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_batch(self) -> "DataBatch":
        if not self.tables:
            raise ValueError("data batch must contain at least one table batch")

        for table_name, table_batch in self.tables.items():
            if table_name != table_batch.table:
                raise ValueError(
                    f"table batch key {table_name} does not match table name "
                    f"{table_batch.table}"
                )

        if self.target and self.target.table not in self.tables:
            raise ValueError(
                f"target table {self.target.table} is not present in data batch"
            )
        return self

    @property
    def primary_table(self) -> TableBatch:
        return next(iter(self.tables.values()))

    @property
    def row_count(self) -> int:
        return self.primary_table.row_count

    @classmethod
    def from_single_table(
        cls,
        table: str,
        columns: List[str],
        rows: Sequence[Sequence[Any]],
        target_column: Optional[str] = None,
        target_values: Optional[Sequence[Any]] = None,
        role: BatchRole = BatchRole.UNKNOWN,
        batch_id: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DataBatch":
        table_batch = TableBatch(table=table, columns=columns, rows=rows)
        target = None
        if target_column is not None:
            if target_values is None:
                raise ValueError("target_values must be provided with target_column")
            target = TargetBatch(
                table=table,
                column=target_column,
                values=target_values,
            )

        return cls(
            tables={table: table_batch},
            target=target,
            role=role,
            batch_id=batch_id,
            session_id=session_id,
            metadata=metadata or {},
        )
