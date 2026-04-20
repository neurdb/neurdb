from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from .base import NonEmptyStr, RuntimeDataModel


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"



class ColumnSchema(RuntimeDataModel):
    dtype: NonEmptyStr
    nullable: bool = True
    role: ColumnRole = ColumnRole.FEATURE
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TableSchema(RuntimeDataModel):
    columns: Dict[NonEmptyStr, ColumnSchema] = Field(min_length=1)
    primary_key: Optional[List[str]] = None
    timestamp_column: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_table(self) -> "TableSchema":
        column_name_set = set(self.columns)

        if self.primary_key and (set(self.primary_key) - column_name_set):
            raise ValueError(
                "table schema primary key columns are missing: "
                f"{set(self.primary_key) - column_name_set}"
            )

        if self.timestamp_column and self.timestamp_column not in column_name_set:
            raise ValueError(
                "table schema timestamp column is missing: "
                f"{self.timestamp_column}"
            )
        return self

    @property
    def column_names(self) -> List[str]:
        return list(self.columns)

    def get_column(self, name: str) -> ColumnSchema:
        try:
            return self.columns[name]
        except KeyError as exc:
            raise KeyError(f"column {name} does not exist in table schema") from exc


class RelationshipSchema(RuntimeDataModel):
    name: NonEmptyStr
    source_table: NonEmptyStr
    source_columns: List[NonEmptyStr] = Field(min_length=1)
    target_table: NonEmptyStr
    target_columns: List[NonEmptyStr] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _validate_column_match(self) -> "RelationshipSchema":
        if len(self.source_columns) != len(self.target_columns):
            raise ValueError(
                f"relationship {self.name} source and target columns must match"
            )
        return self


class DatabaseSchema(RuntimeDataModel):
    tables: Dict[NonEmptyStr, TableSchema] = Field(min_length=1)
    relationships: List[RelationshipSchema] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_database(self) -> "DatabaseSchema":
        
        # mainly check the relationship is valid according to tables
        for r in self.relationships:
            if r.source_table not in self.tables or r.target_table not in self.tables:
                raise ValueError(
                    f"relationship {r.name} source or target table is missing: "
                    f"{r.source_table}, {r.target_table}"
                )

            source_table = self.get_table(r.source_table)
            target_table = self.get_table(r.target_table)
            
            src_missing_columns = set(r.source_columns).issubset(source_table.columns)
            tgt_missing_columns = set(r.target_columns).issubset(target_table.columns)

            if not src_missing_columns or not tgt_missing_columns:
                raise ValueError(
                    f"relationship {r.name} source or target columns are missing: "
                    f"src {r.source_table}: {r.source_columns if not src_missing_columns else ''}, "
                    f"tgt {r.target_table}: {r.target_columns if not tgt_missing_columns else ''}"
                )
        return self


