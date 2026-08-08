import pyarrow as pa
from data.batch import DataBatch
from data.schema import (
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
    TableSchema,
)
from graph.relation import RelationConverter


def _schema(relationships) -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            "orders": TableSchema(
                columns={
                    "order_id": ColumnSchema(role="primary_key"),
                    "user_id": ColumnSchema(role="foreign_key"),
                    "region": ColumnSchema(role="foreign_key"),
                }
            ),
            "users": TableSchema(
                columns={
                    "user_id": ColumnSchema(role="primary_key"),
                    "region": ColumnSchema(role="metadata"),
                }
            ),
        },
        relationships=relationships,
    )


def _fk_orders_users() -> RelationshipSchema:
    return RelationshipSchema(
        name="orders_to_users",
        source_table="orders",
        source_columns=["user_id"],
        target_table="users",
        target_columns=["user_id"],
    )


def _batch(orders: dict, users: dict = None) -> DataBatch:
    tables = {"orders": pa.RecordBatch.from_pydict(orders)}
    if users is not None:
        tables["users"] = pa.RecordBatch.from_pydict(users)
    return DataBatch(tables=tables)


def test_single_column_fk_join_produces_row_index_pairs() -> None:
    converter = RelationConverter(schema=_schema([_fk_orders_users()]))

    graph = converter.construct(
        _batch(
            orders={"order_id": [10, 11, 12], "user_id": [2, 1, 2]},
            users={"user_id": [1, 2], "region": ["sg", "us"]},
        )
    )

    src, tgt = graph.edges["orders_to_users"]
    assert src.tolist() == [0, 1, 2]
    assert tgt.tolist() == [1, 0, 1]


def test_node_counts_reflect_batch_rows_not_schema() -> None:
    converter = RelationConverter(schema=_schema([_fk_orders_users()]))

    graph = converter.construct(
        _batch(
            orders={"order_id": [10], "user_id": [1]},
            users={"user_id": [1, 2, 3], "region": ["a", "b", "c"]},
        )
    )

    assert graph.node_counts == {"orders": 1, "users": 3}


def test_source_rows_without_match_are_dropped() -> None:
    # Inner-join semantics: an order pointing at a user absent from the
    # batch simply contributes no edge.
    converter = RelationConverter(schema=_schema([_fk_orders_users()]))

    graph = converter.construct(
        _batch(
            orders={"order_id": [10, 11], "user_id": [1, 99]},
            users={"user_id": [1], "region": ["sg"]},
        )
    )

    src, tgt = graph.edges["orders_to_users"]
    assert src.tolist() == [0]
    assert tgt.tolist() == [0]


def test_relationship_skipped_when_table_missing_from_batch() -> None:
    # Projection pushed down by the engine decides what's present; a
    # relationship with an absent side is skipped, not an error.
    converter = RelationConverter(schema=_schema([_fk_orders_users()]))

    graph = converter.construct(_batch(orders={"order_id": [10], "user_id": [1]}))

    assert graph.edges == {}
    assert graph.node_counts == {"orders": 1}


def test_composite_key_join_matches_all_columns() -> None:
    rel = RelationshipSchema(
        name="orders_to_users_region",
        source_table="orders",
        source_columns=["user_id", "region"],
        target_table="users",
        target_columns=["user_id", "region"],
    )
    converter = RelationConverter(schema=_schema([rel]))

    graph = converter.construct(
        _batch(
            orders={
                "order_id": [10, 11],
                "user_id": [1, 1],
                "region": ["sg", "us"],  # only (1, "sg") exists on the target
            },
            users={"user_id": [1, 2], "region": ["sg", "us"]},
        )
    )

    src, tgt = graph.edges["orders_to_users_region"]
    assert src.tolist() == [0]
    assert tgt.tolist() == [0]


def test_duplicate_target_keys_keep_last_occurrence() -> None:
    # Pins current behavior: the target index map is built positionally, so
    # a duplicated key resolves to its last row. Acceptable because the
    # target side is the PK side, where keys are unique by construction.
    converter = RelationConverter(schema=_schema([_fk_orders_users()]))

    graph = converter.construct(
        _batch(
            orders={"order_id": [10], "user_id": [1]},
            users={"user_id": [1, 1], "region": ["old", "new"]},
        )
    )

    src, tgt = graph.edges["orders_to_users"]
    assert src.tolist() == [0]
    assert tgt.tolist() == [1]
