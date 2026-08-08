import pyarrow as pa
import pytest
from data.batch import DataBatch
from data.schema import (
    ColumnSchema,
    DatabaseSchema,
    RelationshipSchema,
    TableSchema,
)
from pipeline.view.builder import SqueezedViewBuilder
from pipeline.input import InputPipeline


def _click_schema() -> DatabaseSchema:
    """user-click-item: clicks has its own feature (dwell) but the modeler
    may still choose to treat clicks as pure user->item edges."""
    return DatabaseSchema(
        tables={
            "users": TableSchema(
                columns={
                    "uid": ColumnSchema(role="primary_key"),
                    "age": ColumnSchema(
                        role="feature",
                        metadata={"stype": "numerical", "stats": {}},
                    ),
                }
            ),
            "items": TableSchema(
                columns={
                    "iid": ColumnSchema(role="primary_key"),
                    "price": ColumnSchema(
                        role="feature",
                        metadata={"stype": "numerical", "stats": {}},
                    ),
                }
            ),
            "clicks": TableSchema(
                columns={
                    "uid": ColumnSchema(role="foreign_key"),
                    "iid": ColumnSchema(role="foreign_key"),
                    "dwell": ColumnSchema(
                        role="feature",
                        metadata={"stype": "numerical", "stats": {}},
                    ),
                }
            ),
        },
        relationships=[
            RelationshipSchema(
                name="clicks_to_users",
                source_table="clicks",
                source_columns=["uid"],
                target_table="users",
                target_columns=["uid"],
            ),
            RelationshipSchema(
                name="clicks_to_items",
                source_table="clicks",
                source_columns=["iid"],
                target_table="items",
                target_columns=["iid"],
            ),
        ],
    )


def _click_batch() -> DataBatch:
    return DataBatch(
        tables={
            "users": pa.RecordBatch.from_pydict({"uid": [1, 2], "age": [20.0, 30.0]}),
            "items": pa.RecordBatch.from_pydict({"iid": [7, 8], "price": [1.0, 2.0]}),
            "clicks": pa.RecordBatch.from_pydict(
                {"uid": [1, 2], "iid": [8, 7], "dwell": [3.5, 0.5]}
            ),
        }
    )


def test_default_strategy_keeps_feature_bearing_clicks_as_nodes() -> None:
    # Default rule: clicks has a feature column, so it is NOT squeezed —
    # it stays a node type and its features are encoded.
    pipe = InputPipeline(schema=_click_schema())

    out = pipe.convert(_click_batch())

    assert ("clicks", "dwell") in out.features.features
    assert "clicks" in out.graph.node_counts
    assert set(out.graph.edges) == {"clicks_to_users", "clicks_to_items"}


def test_explicit_squeeze_drops_clicks_from_nodes_and_features() -> None:
    # Strategy override: model clicks as user->item edges. Its features
    # leave the model input, and the feature path skips the table entirely
    # because the strategy's feature_schema is the view FeatureConverter gets.
    builder = SqueezedViewBuilder(schema=_click_schema(), squeeze=["clicks"])
    pipe = InputPipeline(schema=_click_schema(), view_builder=builder)

    out = pipe.convert(_click_batch())

    assert set(builder.feature_schema.tables) == {"users", "items"}

    # Features: users/items only — clicks.dwell is gone.
    assert set(out.features.features) == {("users", "age"), ("items", "price")}

    # Graph: clicks vanished as a node type, composed into one edge type.
    assert out.graph.node_counts == {"users": 2, "items": 2}
    src, tgt = out.graph.edges["clicks"]  # users -> items
    assert list(zip(src.tolist(), tgt.tolist())) == [(0, 1), (1, 0)]


def test_explicit_squeeze_validates_structure() -> None:
    with pytest.raises(ValueError, match="not in schema"):
        SqueezedViewBuilder(schema=_click_schema(), squeeze=["nope"])

    with pytest.raises(ValueError, match="exactly two outgoing"):
        SqueezedViewBuilder(schema=_click_schema(), squeeze=["users"])


def test_features_and_graph_stay_row_aligned() -> None:
    # Node i of table t == row i of the batch: feature arrays and node
    # counts must agree on length for every surviving table.
    builder = SqueezedViewBuilder(schema=_click_schema(), squeeze=["clicks"])
    pipe = InputPipeline(schema=_click_schema(), view_builder=builder)

    out = pipe.convert(_click_batch())

    for (table, _col), arr in out.features.features.items():
        assert len(arr) == out.graph.node_counts[table]
