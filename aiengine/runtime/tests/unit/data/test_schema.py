import pytest
from pydantic import ValidationError

from data.schema import (
    ColumnRole,
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
    TableSchema,
)


def test_column_schema_defaults() -> None:
    column = ColumnSchema()

    assert column.role == ColumnRole.FEATURE
    assert column.metadata == {}


def test_column_schema_accepts_role_override() -> None:
    column = ColumnSchema(role=ColumnRole.TARGET)

    assert column.role == ColumnRole.TARGET


def test_column_schema_uses_distinct_metadata_dicts() -> None:
    first = ColumnSchema()
    second = ColumnSchema()

    first.metadata["source"] = "first"

    assert second.metadata == {}


def test_table_schema_accepts_valid_definition() -> None:
    table = TableSchema(
        columns={
            "id": ColumnSchema(role=ColumnRole.PRIMARY_KEY),
            "created_at": ColumnSchema(role=ColumnRole.TIMESTAMP),
        },
        primary_key=["id"],
        timestamp_column="created_at",
    )

    assert set(table.column_names) == {"id", "created_at"}
    assert table.get_column("id").role == ColumnRole.PRIMARY_KEY


def test_table_schema_rejects_empty_columns() -> None:
    with pytest.raises(ValidationError):
        TableSchema(columns={})


def test_table_schema_rejects_empty_column_name() -> None:
    with pytest.raises(ValidationError):
        TableSchema(columns={"": ColumnSchema()})


def test_table_schema_rejects_missing_primary_key_columns() -> None:
    with pytest.raises(ValidationError, match="table schema primary key columns are missing"):
        TableSchema(
            columns={"id": ColumnSchema()},
            primary_key=["missing_id"],
        )


def test_table_schema_rejects_missing_timestamp_column() -> None:
    with pytest.raises(ValidationError, match="table schema timestamp column is missing"):
        TableSchema(
            columns={"id": ColumnSchema()},
            timestamp_column="created_at",
        )


def test_table_get_column_raises_for_missing_name() -> None:
    table = TableSchema(columns={"id": ColumnSchema()})

    with pytest.raises(KeyError, match="column missing does not exist"):
        table.get_column("missing")


def test_relationship_schema_accepts_matching_columns(
    orders_user_relationship: RelationshipSchema,
) -> None:
    relationship = orders_user_relationship
    assert relationship.name == "orders_user_fk"


def test_relationship_schema_rejects_empty_column_lists() -> None:
    with pytest.raises(ValidationError):
        RelationshipSchema(
            name="orders_user_fk",
            source_table="orders",
            source_columns=[],
            target_table="users",
            target_columns=["id"],
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("source_table", ""),
        ("target_table", ""),
        ("source_columns", [""]),
        ("target_columns", [""]),
    ],
)
def test_relationship_schema_rejects_empty_identifiers(
    field_name: str,
    value: object,
) -> None:
    kwargs = {
        "name": "orders_user_fk",
        "source_table": "orders",
        "source_columns": ["user_id"],
        "target_table": "users",
        "target_columns": ["id"],
    }
    kwargs[field_name] = value

    with pytest.raises(ValidationError):
        RelationshipSchema(**kwargs)


def test_relationship_schema_rejects_mismatched_column_counts() -> None:
    with pytest.raises(ValidationError, match="source and target columns must match"):
        RelationshipSchema(
            name="orders_user_fk",
            source_table="orders",
            source_columns=["tenant_id", "user_id"],
            target_table="users",
            target_columns=["id"],
        )


def test_database_schema_accepts_valid_relationship(
    database_schema: DatabaseSchema,
) -> None:
    schema = database_schema
    assert sorted(schema.tables) == ["orders", "users"]
    assert schema.tables["users"].column_names == ["id"]


def test_database_schema_defaults_to_no_relationships() -> None:
    schema = DatabaseSchema(
        tables={"users": TableSchema(columns={"id": ColumnSchema()})}
    )

    assert schema.relationships == []


def test_database_schema_rejects_missing_relationship_table() -> None:
    with pytest.raises(ValidationError, match="source or target table is missing"):
        DatabaseSchema(
            tables={"orders": TableSchema(columns={"user_id": ColumnSchema()})},
            relationships=[
                RelationshipSchema(
                    name="orders_user_fk",
                    source_table="orders",
                    source_columns=["user_id"],
                    target_table="users",
                    target_columns=["id"],
                )
            ],
        )


@pytest.mark.parametrize(
    ("relationship_kwargs", "missing_side"),
    [
        (
            {
                "source_table": "orders",
                "source_columns": ["user_id"],
                "target_table": "users",
                "target_columns": ["id"],
            },
            "src orders",
        ),
        (
            {
                "source_table": "orders",
                "source_columns": ["id"],
                "target_table": "users",
                "target_columns": ["account_id"],
            },
            "tgt users",
        ),
    ],
)
def test_database_schema_rejects_missing_relationship_columns_on_either_side(
    relationship_kwargs: dict[str, object],
    missing_side: str,
) -> None:
    with pytest.raises(ValidationError, match=missing_side):
        DatabaseSchema(
            tables={
                "users": TableSchema(columns={"id": ColumnSchema()}),
                "orders": TableSchema(columns={"id": ColumnSchema()}),
            },
            relationships=[
                RelationshipSchema(
                    name="orders_user_fk",
                    **relationship_kwargs,
                )
            ],
        )


def test_database_schema_rejects_missing_relationship_columns() -> None:
    with pytest.raises(ValidationError, match="source or target columns are missing"):
        DatabaseSchema(
            tables={
                "users": TableSchema(columns={"id": ColumnSchema()}),
                "orders": TableSchema(columns={"id": ColumnSchema()}),
            },
            relationships=[
                RelationshipSchema(
                    name="orders_user_fk",
                    source_table="orders",
                    source_columns=["user_id"],
                    target_table="users",
                    target_columns=["id"],
                )
            ],
        )


def test_database_schema_rejects_empty_tables() -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema(tables={})
