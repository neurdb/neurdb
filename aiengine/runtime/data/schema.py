from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RECOMMENDATION = "recommendation"
    LINK_PREDICTION = "link_prediction" # NOT USED CURRENTLY
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    role: ColumnRole = ColumnRole.FEATURE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("column name must not be empty")
        if not self.dtype:
            raise ValueError(f"column {self.name} dtype must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSchema":
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            nullable=data.get("nullable", True),
            role=ColumnRole(data.get("role", ColumnRole.FEATURE.value)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "role": self.role.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: List[ColumnSchema]
    primary_key: Optional[List[str]] = None
    timestamp_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("table name must not be empty")
        if not self.columns:
            raise ValueError(f"table {self.name} must contain at least one column")

        column_names = [column.name for column in self.columns]
        duplicate_names = sorted(
            {name for name in column_names if column_names.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(
                f"table {self.name} contains duplicate columns: {duplicate_names}"
            )

        column_name_set = set(column_names)
        if self.primary_key:
            missing_keys = [name for name in self.primary_key if name not in column_name_set]
            if missing_keys:
                raise ValueError(
                    f"table {self.name} primary key columns are missing: {missing_keys}"
                )

        if self.timestamp_column and self.timestamp_column not in column_name_set:
            raise ValueError(
                f"table {self.name} timestamp column is missing: "
                f"{self.timestamp_column}"
            )

    @property
    def column_names(self) -> List[str]:
        return [column.name for column in self.columns]

    def get_column(self, name: str) -> ColumnSchema:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"column {name} does not exist in table {self.name}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSchema":
        return cls(
            name=data["name"],
            columns=[ColumnSchema.from_dict(column) for column in data["columns"]],
            primary_key=data.get("primary_key"),
            timestamp_column=data.get("timestamp_column"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [column.to_dict() for column in self.columns],
            "primary_key": list(self.primary_key) if self.primary_key else None,
            "timestamp_column": self.timestamp_column,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RelationshipSchema:
    name: str
    source_table: str
    source_columns: List[str]
    target_table: str
    target_columns: List[str]
    cardinality: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("relationship name must not be empty")
        if not self.source_table or not self.target_table:
            raise ValueError(f"relationship {self.name} table names must not be empty")
        if not self.source_columns or not self.target_columns:
            raise ValueError(f"relationship {self.name} columns must not be empty")
        if len(self.source_columns) != len(self.target_columns):
            raise ValueError(
                f"relationship {self.name} source and target columns must match"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipSchema":
        return cls(
            name=data["name"],
            source_table=data["source_table"],
            source_columns=list(data["source_columns"]),
            target_table=data["target_table"],
            target_columns=list(data["target_columns"]),
            cardinality=data.get("cardinality"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "source_columns": list(self.source_columns),
            "target_table": self.target_table,
            "target_columns": list(self.target_columns),
            "cardinality": self.cardinality,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TargetSpec:
    table: str
    column: str
    task_type: TaskType = TaskType.UNKNOWN
    timestamp_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.table:
            raise ValueError("target table must not be empty")
        if not self.column:
            raise ValueError("target column must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSpec":
        return cls(
            table=data["table"],
            column=data["column"],
            task_type=TaskType(data.get("task_type", TaskType.UNKNOWN.value)),
            timestamp_column=data.get("timestamp_column"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "task_type": self.task_type.value,
            "timestamp_column": self.timestamp_column,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatabaseSchema:
    tables: List[TableSchema]
    relationships: List[RelationshipSchema] = field(default_factory=list)
    target: Optional[TargetSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tables:
            raise ValueError("database schema must contain at least one table")

        table_names = [table.name for table in self.tables]
        duplicate_names = sorted(
            {name for name in table_names if table_names.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(f"database schema contains duplicate tables: {duplicate_names}")

        table_name_set = set(table_names)
        for relationship in self.relationships:
            if relationship.source_table not in table_name_set:
                raise ValueError(
                    f"relationship {relationship.name} source table is missing: "
                    f"{relationship.source_table}"
                )
            if relationship.target_table not in table_name_set:
                raise ValueError(
                    f"relationship {relationship.name} target table is missing: "
                    f"{relationship.target_table}"
                )

            source_table = self.get_table(relationship.source_table)
            target_table = self.get_table(relationship.target_table)
            _validate_columns_exist(
                source_table, relationship.source_columns, relationship.name
            )
            _validate_columns_exist(
                target_table, relationship.target_columns, relationship.name
            )

        if self.target:
            if self.target.table not in table_name_set:
                raise ValueError(f"target table is missing: {self.target.table}")
            target_table = self.get_table(self.target.table)
            target_table.get_column(self.target.column)
            if self.target.timestamp_column:
                target_table.get_column(self.target.timestamp_column)

    def get_table(self, name: str) -> TableSchema:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"table {name} does not exist")

    @classmethod
    def single_table(
        cls,
        table_name: str,
        columns: Iterable[ColumnSchema],
        target_column: Optional[str] = None,
        task_type: TaskType = TaskType.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DatabaseSchema":
        table = TableSchema(name=table_name, columns=list(columns))
        target = (
            TargetSpec(table=table_name, column=target_column, task_type=task_type)
            if target_column
            else None
        )
        return cls(tables=[table], target=target, metadata=metadata or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseSchema":
        target_data = data.get("target")
        return cls(
            tables=[TableSchema.from_dict(table) for table in data["tables"]],
            relationships=[
                RelationshipSchema.from_dict(relationship)
                for relationship in data.get("relationships", [])
            ],
            target=TargetSpec.from_dict(target_data) if target_data else None,
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
            "target": self.target.to_dict() if self.target else None,
            "metadata": dict(self.metadata),
        }


def _validate_columns_exist(
    table: TableSchema, column_names: Iterable[str], relationship_name: str
):
    missing_columns = [name for name in column_names if name not in table.column_names]
    if missing_columns:
        raise ValueError(
            f"relationship {relationship_name} references missing columns on "
            f"table {table.name}: {missing_columns}"
        )
