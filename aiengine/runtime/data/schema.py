from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import Field, model_validator

from .base import ColumnRole, ColumnStype, MetadataKey, NonEmptyStr, RuntimeDataModel


class ColumnSchema(RuntimeDataModel):
    """Per-column schema entry.

    ``metadata`` is a free-form dict used as a tagged-union payload — the DB
    engine populates it because only the engine has the global dataset view
    (a ``DataBatch`` is a subset and cannot derive global statistics).

    Convention consumed by the runtime converter:

    - ``stype`` (str, required when the column reaches the model — i.e. role
      in {FEATURE, TARGET, TIMESTAMP}; omit for PRIMARY_KEY / FOREIGN_KEY /
      METADATA). Discriminator for the encoding path. One of:

        * ``"numerical"``   — scalar float; linear projection / MLP input.
        * ``"categorical"`` — discrete vocab; embedding lookup.
        * ``"timestamp"``   — datetime; cast to int64 epoch.

    - ``stats`` (dict, shape selected by ``stype``):

        numerical   -> {"mean": float, "std": float}
        categorical -> {"cardinality": list[Any]}    # ordered; index == global code
        timestamp   -> {"strategy": "epoch_s" | "epoch_ms" | "epoch_ns"}

    Extra keys (quantiles, class counts, year_range, ...) are permitted and
    ignored by the converter — reserved for future use.
    """

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

    @model_validator(mode="after")
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

    def iter_columns(
        self, role: Optional["ColumnRole"] = None
    ) -> Iterator[Tuple[str, str, ColumnSchema]]:
        """Yield (table_name, column_name, column_schema) across all tables."""
        for table_name, table in self.tables.items():
            for col_name, col in table.columns.items():
                if role is None or col.role is role:
                    yield table_name, col_name, col

    @model_validator(mode="after")
    def _validate_database(self) -> "DatabaseSchema":
        for r in self.relationships:
            if r.source_table not in self.tables or r.target_table not in self.tables:
                raise ValueError(
                    f"relationship {r.name} source or target table is missing: "
                    f"{r.source_table}, {r.target_table}"
                )

            source_table = self.tables[r.source_table]
            target_table = self.tables[r.target_table]

            src_columns_present = set(r.source_columns).issubset(source_table.columns)
            tgt_columns_present = set(r.target_columns).issubset(target_table.columns)

            if not src_columns_present or not tgt_columns_present:
                raise ValueError(
                    f"relationship {r.name} source or target columns are missing: "
                    f"src {r.source_table}: {r.source_columns if not src_columns_present else ''}, "
                    f"tgt {r.target_table}: {r.target_columns if not tgt_columns_present else ''}"
                )
        return self
