from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class BatchRole(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    TEST = "test"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TableBatch:
    table: str
    columns: List[str]
    rows: Sequence[Sequence[Any]]
    row_ids: Optional[Sequence[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.table:
            raise ValueError("table batch table name must not be empty")
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

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableBatch":
        return cls(
            table=data["table"],
            columns=list(data["columns"]),
            rows=list(data.get("rows", [])),
            row_ids=data.get("row_ids"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "row_ids": list(self.row_ids) if self.row_ids is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TargetBatch:
    table: str
    column: str
    values: Sequence[Any]
    row_ids: Optional[Sequence[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.table:
            raise ValueError("target batch table name must not be empty")
        if not self.column:
            raise ValueError("target batch column must not be empty")
        if self.row_ids is not None and len(self.row_ids) != len(self.values):
            raise ValueError("target batch row_ids length must match values length")

    @property
    def row_count(self) -> int:
        return len(self.values)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetBatch":
        return cls(
            table=data["table"],
            column=data["column"],
            values=list(data.get("values", [])),
            row_ids=data.get("row_ids"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "values": list(self.values),
            "row_ids": list(self.row_ids) if self.row_ids is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DataBatch:
    tables: Mapping[str, TableBatch]
    target: Optional[TargetBatch] = None
    role: BatchRole = BatchRole.UNKNOWN
    batch_id: Optional[int] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataBatch":
        tables = {
            name: TableBatch.from_dict(table_data)
            for name, table_data in data["tables"].items()
        }
        target_data = data.get("target")
        return cls(
            tables=tables,
            target=TargetBatch.from_dict(target_data) if target_data else None,
            role=BatchRole(data.get("role", BatchRole.UNKNOWN.value)),
            batch_id=data.get("batch_id"),
            session_id=data.get("session_id"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": {
                name: table_batch.to_dict() for name, table_batch in self.tables.items()
            },
            "target": self.target.to_dict() if self.target else None,
            "role": self.role.value,
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "metadata": dict(self.metadata),
        }
