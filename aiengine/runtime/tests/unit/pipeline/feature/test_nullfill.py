import pyarrow as pa
import pytest
from data.schema import ColumnSchema
from pipeline.feature.base import Identity
from pipeline.feature.nullfill import (
    NullFillBackward,
    NullFillBuilder,
    NullFillConstant,
    NullFillForward,
)


def _column(stype: str, stats: dict = None) -> ColumnSchema:
    metadata = {"stype": stype}
    if stats is not None:
        metadata["stats"] = stats
    return ColumnSchema(role="feature", metadata=metadata)


# ---------------------------------------------------------------------------
# NullFillBuilder wiring
# ---------------------------------------------------------------------------


def test_builder_numerical_with_impute_builds_constant_fill() -> None:
    op = NullFillBuilder()(_column("numerical", {"mean": 2.0, "impute": 2.0}))

    assert isinstance(op, NullFillConstant)
    assert op.value == 2.0


def test_builder_numerical_without_impute_does_not_apply() -> None:
    # The runtime never invents statistics: no engine-provided impute value
    # means no fill step, and nulls flow through to the encoder.
    op = NullFillBuilder()(_column("numerical", {"mean": 2.0, "std": 1.0}))

    assert op is None


def test_builder_categorical_builds_forward_fill() -> None:
    op = NullFillBuilder()(_column("categorical", {"cardinality": ["a"]}))

    assert isinstance(op, NullFillForward)


def test_builder_timestamp_builds_identity() -> None:
    op = NullFillBuilder()(_column("timestamp"))

    assert isinstance(op, Identity)


def test_builder_no_stype_does_not_apply() -> None:
    op = NullFillBuilder()(ColumnSchema(role="primary_key"))

    assert op is None


def test_register_rejects_duplicate_strategy_name() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @NullFillBuilder.register("forward")
        class Duplicate:
            def apply(self, arr: pa.Array) -> pa.Array:
                return arr


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_constant_fill_replaces_nulls_with_value() -> None:
    out = NullFillConstant(value=9.0).apply(pa.array([1.0, None, 3.0]))

    assert out.to_pylist() == [1.0, 9.0, 3.0]


def test_forward_fill_carries_last_value_and_keeps_leading_null() -> None:
    out = NullFillForward().apply(pa.array(["x", None, "y", None]))

    assert out.to_pylist() == ["x", "x", "y", "y"]

    leading = NullFillForward().apply(pa.array([None, "x"]))
    assert leading.to_pylist() == [None, "x"]


def test_backward_fill_carries_next_value_and_keeps_trailing_null() -> None:
    out = NullFillBackward().apply(pa.array([None, "x", None]))

    assert out.to_pylist() == ["x", "x", None]
