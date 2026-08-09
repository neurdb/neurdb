import pyarrow as pa
from data.batch import DataBatch
from data.schema import (
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
    TableSchema,
)
from pipeline.view.relation import RelationConverter
from pipeline.view.squeeze import RelationSqueezer


def _m2m_schema(junction_extra_columns=None) -> DatabaseSchema:
    junction_columns = {
        "order_id": ColumnSchema(role="foreign_key"),
        "product_id": ColumnSchema(role="foreign_key"),
    }
    junction_columns.update(junction_extra_columns or {})
    return DatabaseSchema(
        tables={
            "orders": TableSchema(
                columns={"order_id": ColumnSchema(role="primary_key")}
            ),
            "products": TableSchema(
                columns={"product_id": ColumnSchema(role="primary_key")}
            ),
            "order_products": TableSchema(columns=junction_columns),
        },
        relationships=[
            RelationshipSchema(
                name="op_to_orders",
                source_table="order_products",
                source_columns=["order_id"],
                target_table="orders",
                target_columns=["order_id"],
            ),
            RelationshipSchema(
                name="op_to_products",
                source_table="order_products",
                source_columns=["product_id"],
                target_table="products",
                target_columns=["product_id"],
            ),
        ],
    )


def _m2m_batch(op_rows: dict) -> DataBatch:
    return DataBatch(
        tables={
            "orders": pa.RecordBatch.from_pydict({"order_id": [10, 11]}),
            "products": pa.RecordBatch.from_pydict({"product_id": [1, 2, 3]}),
            "order_products": pa.RecordBatch.from_pydict(op_rows),
        }
    )


def test_junction_squeezed_into_direct_edge() -> None:
    schema = _m2m_schema()
    graph = RelationConverter(schema=schema).construct(
        _m2m_batch({"order_id": [10, 10, 11], "product_id": [1, 3, 2]})
    )

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    # Junction vanished as a node type; one composed edge type remains,
    # named after the junction table.
    assert squeezed.node_counts == {"orders": 2, "products": 3}
    assert set(squeezed.edges) == {"order_products"}

    src, tgt = squeezed.edges["order_products"]  # orders -> products
    assert list(zip(src.tolist(), tgt.tolist())) == [(0, 0), (0, 2), (1, 1)]


def test_junction_with_feature_column_stays_a_node() -> None:
    # A junction with its own semantics (e.g. a rating) must not be
    # squeezed: it has features the model needs.
    schema = _m2m_schema(
        junction_extra_columns={
            "rating": ColumnSchema(
                role="feature",
                metadata={"stype": "numerical", "stats": {}},
            )
        }
    )
    graph = RelationConverter(schema=schema).construct(
        _m2m_batch({"order_id": [10], "product_id": [1], "rating": [5.0]})
    )

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    assert "order_products" in squeezed.node_counts
    assert set(squeezed.edges) == {"op_to_orders", "op_to_products"}


def test_junction_absent_from_batch_leaves_graph_unchanged() -> None:
    schema = _m2m_schema()
    batch = DataBatch(
        tables={
            "orders": pa.RecordBatch.from_pydict({"order_id": [10]}),
            "products": pa.RecordBatch.from_pydict({"product_id": [1]}),
        }
    )
    graph = RelationConverter(schema=schema).construct(batch)

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    assert squeezed.node_counts == graph.node_counts
    assert squeezed.edges == graph.edges


def test_entity_table_absent_keeps_junction_as_node() -> None:
    # products missing from the batch -> op_to_products edge never built ->
    # composition impossible; the junction passes through untouched.
    schema = _m2m_schema()
    batch = DataBatch(
        tables={
            "orders": pa.RecordBatch.from_pydict({"order_id": [10]}),
            "order_products": pa.RecordBatch.from_pydict(
                {"order_id": [10], "product_id": [1]}
            ),
        }
    )
    graph = RelationConverter(schema=schema).construct(batch)

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    assert "order_products" in squeezed.node_counts
    assert set(squeezed.edges) == {"op_to_orders"}


def test_null_fk_row_produces_no_composed_edge() -> None:
    schema = _m2m_schema()
    graph = RelationConverter(schema=schema).construct(
        _m2m_batch({"order_id": [10, 11], "product_id": [1, None]})
    )

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    src, tgt = squeezed.edges["order_products"]
    assert list(zip(src.tolist(), tgt.tolist())) == [(0, 0)]


def test_self_relation_junction_squeezes_within_one_table() -> None:
    schema = DatabaseSchema(
        tables={
            "users": TableSchema(columns={"uid": ColumnSchema(role="primary_key")}),
            "friendships": TableSchema(
                columns={
                    "uid_a": ColumnSchema(role="foreign_key"),
                    "uid_b": ColumnSchema(role="foreign_key"),
                }
            ),
        },
        relationships=[
            RelationshipSchema(
                name="f_to_a",
                source_table="friendships",
                source_columns=["uid_a"],
                target_table="users",
                target_columns=["uid"],
            ),
            RelationshipSchema(
                name="f_to_b",
                source_table="friendships",
                source_columns=["uid_b"],
                target_table="users",
                target_columns=["uid"],
            ),
        ],
    )
    batch = DataBatch(
        tables={
            "users": pa.RecordBatch.from_pydict({"uid": [1, 2, 3]}),
            "friendships": pa.RecordBatch.from_pydict(
                {"uid_a": [1, 2], "uid_b": [2, 3]}
            ),
        }
    )
    graph = RelationConverter(schema=schema).construct(batch)

    squeezed = RelationSqueezer(schema=schema).apply(graph)

    assert squeezed.node_counts == {"users": 3}
    src, tgt = squeezed.edges["friendships"]  # users -> users
    assert list(zip(src.tolist(), tgt.tolist())) == [(0, 1), (1, 2)]


def test_targeted_junction_is_not_squeezed() -> None:
    # Another table FKs into the junction -> squeezing would leave that
    # edge dangling, so the junction must stay a node type.
    schema = _m2m_schema()
    tables = dict(schema.tables)
    tables["op_audit"] = TableSchema(
        columns={"op_row": ColumnSchema(role="foreign_key")}
    )
    tables["order_products"] = TableSchema(
        columns={
            "op_id": ColumnSchema(role="primary_key"),
            "order_id": ColumnSchema(role="foreign_key"),
            "product_id": ColumnSchema(role="foreign_key"),
        }
    )
    schema = DatabaseSchema(
        tables=tables,
        relationships=list(schema.relationships)
        + [
            RelationshipSchema(
                name="audit_to_op",
                source_table="op_audit",
                source_columns=["op_row"],
                target_table="order_products",
                target_columns=["op_id"],
            )
        ],
    )

    assert RelationSqueezer(schema=schema)._plan == []
