import numpy as np
import pyarrow as pa
import pytest
from pipeline.feature.encoder import (
    CategoricalEncoder,
    NumericEncoder,
    TimestampEncoder,
)

# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------


def test_categorical_encoder_maps_values_to_list_position() -> None:
    encoder = CategoricalEncoder(cardinality=["a", "b", "c"])

    out = encoder.encode(pa.array(["b", "a", "c", "a"]))

    assert out.tolist() == [1, 0, 2, 0]


def test_categorical_encoder_returns_int64() -> None:
    encoder = CategoricalEncoder(cardinality=["x", "y"])

    out = encoder.encode(pa.array(["x", "y", "x"]))

    assert out.dtype == np.int64


def test_categorical_encoder_handles_empty_input() -> None:
    encoder = CategoricalEncoder(cardinality=["a", "b"])

    out = encoder.encode(pa.array([], type=pa.utf8()))

    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_categorical_encoder_supports_integer_cardinality() -> None:
    # The encoder is generic over arrow types; it just maps values to positions.
    encoder = CategoricalEncoder(cardinality=[10, 20, 30])

    out = encoder.encode(pa.array([20, 30, 10, 30]))

    assert out.tolist() == [1, 2, 0, 2]


def test_categorical_encoder_unknown_value_maps_to_reserved_code() -> None:
    # Out-of-vocabulary values get the reserved code len(cardinality) — the
    # last row of a (cardinality + 1)-sized embedding table.
    encoder = CategoricalEncoder(cardinality=["a", "b"])

    out = encoder.encode(pa.array(["a", "zzz"]))

    assert out.dtype == np.int64
    assert encoder.unknown_code == 2
    assert out.tolist() == [0, 2]


def test_categorical_encoder_null_maps_to_reserved_code() -> None:
    # Nulls that survive null-fill (e.g. leading nulls under forward-fill)
    # land on the same reserved code as unknown values.
    encoder = CategoricalEncoder(cardinality=["a", "b"])

    out = encoder.encode(pa.array(["b", None]))

    assert out.tolist() == [1, encoder.unknown_code]


# ---------------------------------------------------------------------------
# NumericEncoder
# ---------------------------------------------------------------------------


def test_numeric_encoder_default_passes_through_to_float32() -> None:
    encoder = NumericEncoder()

    out = encoder.encode(pa.array([1.0, 2.0, 3.0]))

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_numeric_encoder_casts_integer_input_to_float32() -> None:
    encoder = NumericEncoder()

    out = encoder.encode(pa.array([1, 2, 3], type=pa.int64()))

    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_numeric_encoder_standardizes_when_mean_or_std_set() -> None:
    encoder = NumericEncoder(mean=2.0, std=2.0)

    out = encoder.encode(pa.array([2.0, 4.0, 6.0]))

    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])


def test_numeric_encoder_zero_std_is_treated_as_unit_divisor() -> None:
    # std=0 with a non-zero mean: the implementation guards via `self.std or 1.0`
    # so the result is just (x - mean), no div-by-zero.
    encoder = NumericEncoder(mean=1.0, std=0.0)

    out = encoder.encode(pa.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])


def test_numeric_encoder_handles_empty_input() -> None:
    encoder = NumericEncoder(mean=5.0, std=2.0)

    out = encoder.encode(pa.array([], type=pa.float64()))

    assert out.shape == (0,)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# TimestampEncoder
# ---------------------------------------------------------------------------


def test_timestamp_encoder_default_returns_epoch_seconds() -> None:
    encoder = TimestampEncoder()

    out = encoder.encode(
        pa.array([1_700_000_000, 1_700_000_001], type=pa.timestamp("s"))
    )

    assert out.dtype == np.int64
    assert out.tolist() == [1_700_000_000, 1_700_000_001]


def test_timestamp_encoder_epoch_ms_scales_seconds_input() -> None:
    encoder = TimestampEncoder(strategy="epoch_ms")

    out = encoder.encode(pa.array([1_700_000_000], type=pa.timestamp("s")))

    assert out.tolist() == [1_700_000_000_000]


def test_timestamp_encoder_epoch_ns_passthrough_for_ns_input() -> None:
    encoder = TimestampEncoder(strategy="epoch_ns")

    out = encoder.encode(pa.array([1_700_000_000_000_000_000], type=pa.timestamp("ns")))

    assert out.dtype == np.int64
    assert out.tolist() == [1_700_000_000_000_000_000]


def test_timestamp_encoder_rejects_non_timestamp_input() -> None:
    encoder = TimestampEncoder()

    with pytest.raises(TypeError, match="expected timestamp"):
        encoder.encode(pa.array([1, 2, 3], type=pa.int64()))


def test_timestamp_encoder_handles_empty_input() -> None:
    encoder = TimestampEncoder(strategy="epoch_s")

    out = encoder.encode(pa.array([], type=pa.timestamp("s")))

    assert out.shape == (0,)
    assert out.dtype == np.int64
