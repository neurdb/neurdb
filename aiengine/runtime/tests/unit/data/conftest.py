import pytest

from data.schema import (
    ColumnRole,
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
    TableSchema,
)


@pytest.fixture
def users_table() -> TableSchema:
    return TableSchema(
        columns={"id": ColumnSchema(role=ColumnRole.PRIMARY_KEY)},
        primary_key=["id"],
    )


@pytest.fixture
def orders_table() -> TableSchema:
    return TableSchema(
        columns={
            "id": ColumnSchema(role=ColumnRole.PRIMARY_KEY),
            "user_id": ColumnSchema(role=ColumnRole.FOREIGN_KEY),
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
