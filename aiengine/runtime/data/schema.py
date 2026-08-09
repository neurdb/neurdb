from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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
                "table schema timestamp column is missing: " f"{self.timestamp_column}"
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


class FeatureSchema(RuntimeDataModel):
    """Slim, derived view of what the feature path should encode.

    Produced from a ``DatabaseSchema`` by a ``ViewBuilder`` at job setup —
    deliberately *not* a subclass: the database-level metainformation
    (relationships, primary keys, timestamp columns, cross-validation) does
    not apply to a derived view and must not tag along. All that feature
    encoding needs is which columns exist and their ``ColumnSchema``.

    Like every runtime data model it is frozen — a static artifact of job
    configuration, not a status variable. Strategies that synthesize
    columns (e.g. DFS aggregates) derive a new instance via ``extended``.
    """

    tables: Dict[NonEmptyStr, Dict[NonEmptyStr, ColumnSchema]]

    @classmethod
    def from_database(
        cls,
        schema: "DatabaseSchema",
        tables: Optional[Iterable[str]] = None,
    ) -> "FeatureSchema":
        """Project a DatabaseSchema down to a feature view.

        ``tables`` selects which tables survive (default: all).
        """
        selected = set(schema.tables) if tables is None else set(tables)
        return cls(
            tables={
                name: dict(table.columns)
                for name, table in schema.tables.items()
                if name in selected
            }
        )

    def get_column(self, table: str, column: str) -> Optional[ColumnSchema]:
        columns = self.tables.get(table)
        return None if columns is None else columns.get(column)

    def extended(self, table: str, columns: Dict[str, ColumnSchema]) -> "FeatureSchema":
        """A new view with extra (synthesized) columns on ``table``."""
        tables = {name: dict(cols) for name, cols in self.tables.items()}
        tables.setdefault(table, {}).update(columns)
        return FeatureSchema(tables=tables)


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
