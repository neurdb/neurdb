import pytest

from data.schema import ColumnSchema, DatabaseSchema, RelationshipSchema, TableSchema


@pytest.fixture
def int_column() -> ColumnSchema:
    return ColumnSchema(dtype="int")


@pytest.fixture
def timestamp_column() -> ColumnSchema:
    return ColumnSchema(dtype="timestamp")


@pytest.fixture
def users_table(int_column: ColumnSchema) -> TableSchema:
    return TableSchema(columns={"id": int_column}, primary_key=["id"])


@pytest.fixture
def orders_table(int_column: ColumnSchema) -> TableSchema:
    return TableSchema(
        columns={
            "id": int_column,
            "user_id": ColumnSchema(dtype="int"),
        }
    )


@pytest.fixture
def orders_user_relationship() -> RelationshipSchema:
    return RelationshipSchema(
        name="orders_user_fk",
        source_table="orders",
        source_columns=["user_id"],
        target_table="users",
        target_columns=["id"],
    )


@pytest.fixture
def database_schema(
    users_table: TableSchema,
    orders_table: TableSchema,
    orders_user_relationship: RelationshipSchema,
) -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            "users": users_table,
            "orders": orders_table,
        },
        relationships=[orders_user_relationship],
    )
