import pyarrow as pa
import pytest
from pydantic import ValidationError

from data.batch import DataBatch


def make_users_batch() -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(["alice", "bob", "carol"], type=pa.utf8()),
        ],
        names=["id", "name"],
    )


def make_orders_batch() -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array([10, 11, 12], type=pa.int64()),
            pa.array([1, 2, 1], type=pa.int64()),
        ],
        names=["id", "user_id"],
    )


def test_data_batch_accepts_single_table() -> None:
    users = make_users_batch()

    batch = DataBatch(tables={"users": users}, anchor_table="users")

    assert batch.anchor_table == "users"
    assert batch.tables["users"].num_rows == 3
    assert batch.metadata == {}


def test_data_batch_accepts_multiple_tables() -> None:
    batch = DataBatch(
        tables={"users": make_users_batch(), "orders": make_orders_batch()},
        anchor_table="users",
    )

    assert set(batch.tables) == {"users", "orders"}
    assert batch.tables["orders"].num_rows == 3


def test_data_batch_rejects_empty_tables() -> None:
    with pytest.raises(ValidationError, match="must contain at least one table"):
        DataBatch(tables={}, anchor_table="users")


def test_data_batch_rejects_missing_anchor_table() -> None:
    with pytest.raises(ValidationError, match="anchor table orders"):
        DataBatch(tables={"users": make_users_batch()}, anchor_table="orders")


def test_data_batch_rejects_empty_anchor_table() -> None:
    with pytest.raises(ValidationError):
        DataBatch(tables={"users": make_users_batch()}, anchor_table="")


def test_data_batch_roundtrip_preserves_scalar_fields() -> None:
    original = DataBatch(
        tables={"users": make_users_batch()},
        anchor_table="users",
        metadata={"role": "train"},
    )

    restored = DataBatch.from_bytes(original.to_bytes())

    assert restored.anchor_table == "users"
    assert restored.metadata == {"role": "train"}


def test_data_batch_roundtrip_preserves_arrow_payload() -> None:
    users = make_users_batch()
    original = DataBatch(tables={"users": users}, anchor_table="users")

    restored = DataBatch.from_bytes(original.to_bytes())

    rb = restored.tables["users"]
    assert rb.num_rows == 3
    assert rb.schema.equals(users.schema)
    assert rb.column("id").to_pylist() == [1, 2, 3]
    assert rb.column("name").to_pylist() == ["alice", "bob", "carol"]


def test_data_batch_roundtrip_preserves_multiple_tables() -> None:
    original = DataBatch(
        tables={"users": make_users_batch(), "orders": make_orders_batch()},
        anchor_table="users",
    )

    restored = DataBatch.from_bytes(original.to_bytes())

    assert set(restored.tables) == {"users", "orders"}
    assert restored.tables["orders"].column("user_id").to_pylist() == [1, 2, 1]


def test_data_batch_roundtrip_preserves_dictionary_column() -> None:
    users = pa.record_batch(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(
                ["premium", "basic", "premium"],
                type=pa.dictionary(pa.int32(), pa.utf8()),
            ),
        ],
        names=["id", "tier"],
    )
    original = DataBatch(tables={"users": users}, anchor_table="users")

    restored = DataBatch.from_bytes(original.to_bytes())

    tier = restored.tables["users"].column("tier")
    assert tier.dictionary.to_pylist() == ["premium", "basic"]
    assert tier.indices.to_pylist() == [0, 1, 0]


def test_data_batch_from_bytes_rejects_invalid_magic() -> None:
    with pytest.raises(ValueError, match="magic header missing"):
        DataBatch.from_bytes(b"NOTNEURDB" + b"\x00" * 16)
