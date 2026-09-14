import logging

import numpy as np
import pyarrow as pa
import pytest
from data.batch import DataBatch
from data.schema import ColumnSchema, DatabaseSchema, FeatureSchema, TableSchema
from pipeline.feature.base import EncodeError
from pipeline.feature.converter import FeatureConverter


def _converter(db_schema: DatabaseSchema) -> FeatureConverter:
    return FeatureConverter(schema=FeatureSchema.from_database(db_schema))


# ---------------------------------------------------------------------------
# End-to-end: default builders over all three stypes
# ---------------------------------------------------------------------------


def _users_schema(**column_overrides) -> DatabaseSchema:
    columns = {
        "id": ColumnSchema(role="primary_key"),
        "age": ColumnSchema(
            role="feature",
            metadata={
                "stype": "numerical",
                "stats": {"mean": 30.0, "std": 10.0, "impute": 30.0},
            },
        ),
        "country": ColumnSchema(
            role="feature",
            metadata={
                "stype": "categorical",
                "stats": {"cardinality": ["sg", "us"]},
            },
        ),
        "joined_at": ColumnSchema(
            role="timestamp",
            metadata={"stype": "timestamp", "stats": {"strategy": "epoch_s"}},
        ),
    }
    columns.update(column_overrides)
    return DatabaseSchema(tables={"users": TableSchema(columns=columns)})


def _users_batch() -> DataBatch:
    rb = pa.RecordBatch.from_pydict(
        {
            "id": pa.array([1, 2, 3, 4]),
            "age": pa.array([20.0, None, 40.0, 50.0]),
            "country": pa.array(["sg", "us", None, "jp"]),
            "joined_at": pa.array(
                [1_700_000_000, 1_700_000_001, 1_700_000_002, 1_700_000_003],
                type=pa.timestamp("s"),
            ),
        }
    )
    return DataBatch(tables={"users": rb})


def test_converter_encodes_all_three_stypes_with_default_builders() -> None:
    converter = _converter(_users_schema())

    out = converter.apply(_users_batch())

    # Primary key has no stype and is skipped; the three model columns encode.
    assert set(out.features) == {
        ("users", "age"),
        ("users", "country"),
        ("users", "joined_at"),
    }

    # Numerical: null imputed with 30.0, then standardized with mean/std.
    np.testing.assert_allclose(out.features[("users", "age")], [-1.0, 0.0, 1.0, 2.0])

    # Categorical: null forward-filled to "us" (code 1), unknown "jp" -> 2.
    assert out.features[("users", "country")].tolist() == [0, 1, 1, 2]

    # Timestamp: epoch seconds as int64.
    assert out.features[("users", "joined_at")].tolist() == [
        1_700_000_000,
        1_700_000_001,
        1_700_000_002,
        1_700_000_003,
    ]


def test_converter_reuses_cached_pipeline_across_batches() -> None:
    converter = _converter(_users_schema())

    converter.apply(_users_batch())
    first = dict(converter._cache)
    converter.apply(_users_batch())

    assert dict(converter._cache) == first  # same objects, built once


# ---------------------------------------------------------------------------
# Dropped-column reporting
# ---------------------------------------------------------------------------


def test_dropped_feature_column_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    # A FEATURE column whose metadata cannot build a pipeline is a schema
    # bug: warn (once), don't silently drop.
    schema = _users_schema(country=ColumnSchema(role="feature"))  # no stype
    converter = _converter(schema)

    with caplog.at_level(logging.WARNING, logger="pipeline.feature.converter"):
        converter.apply(_users_batch())
        converter.apply(_users_batch())

    warnings = [r for r in caplog.records if "users.country" in r.message]
    assert len(warnings) == 1
    assert "feature" in warnings[0].message


def test_dropped_key_column_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    converter = _converter(_users_schema())

    with caplog.at_level(logging.WARNING, logger="pipeline.feature.converter"):
        converter.apply(_users_batch())

    assert not [r for r in caplog.records if "users.id" in r.message]


# ---------------------------------------------------------------------------
# Error context
# ---------------------------------------------------------------------------


def test_encode_failure_names_the_offending_column() -> None:
    # Schema says timestamp but the batch delivers int64 -> the encoder's
    # TypeError surfaces as EncodeError naming users.joined_at.
    schema = _users_schema()
    rb = pa.RecordBatch.from_pydict({"joined_at": pa.array([1, 2, 3], type=pa.int64())})
    converter = _converter(schema)

    with pytest.raises(EncodeError, match=r"users\.joined_at"):
        converter.apply(DataBatch(tables={"users": rb}))
