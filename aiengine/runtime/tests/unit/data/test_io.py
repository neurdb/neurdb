import io

import pyarrow as pa

from data.batch import DataBatch
from data.io import StreamDataSource, encode_frame, encode_stream


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
            pa.array([10, 11], type=pa.int64()),
            pa.array([1, 2], type=pa.int64()),
        ],
        names=["id", "user_id"],
    )


def test_stream_roundtrip_yields_frames_in_order() -> None:
    frame_a = DataBatch(tables={"users": make_users_batch()})
    frame_b = DataBatch(
        tables={"users": make_users_batch(), "orders": make_orders_batch()},
    )

    wire = encode_stream([frame_a, frame_b])
    restored = list(StreamDataSource(io.BytesIO(wire)))

    assert len(restored) == 2
    assert set(restored[0].tables) == {"users"}
    assert set(restored[1].tables) == {"users", "orders"}
    assert restored[1].tables["orders"].column("user_id").to_pylist() == [1, 2]


def test_stream_preserves_arrow_payload_across_frames() -> None:
    original = DataBatch(tables={"users": make_users_batch()})

    wire = encode_stream([original])
    (restored,) = list(StreamDataSource(io.BytesIO(wire)))

    rb = restored.tables["users"]
    assert rb.column("id").to_pylist() == [1, 2, 3]
    assert rb.column("name").to_pylist() == ["alice", "bob", "carol"]


def test_empty_stream_yields_nothing() -> None:
    wire = encode_stream([])

    assert list(StreamDataSource(io.BytesIO(wire))) == []


def test_terminal_frame_stops_iteration_before_trailing_bytes() -> None:
    wire = encode_stream([DataBatch(tables={"users": make_users_batch()})])

    # Junk after the terminal frame must never be read.
    src = StreamDataSource(io.BytesIO(wire + b"\xde\xad\xbe\xef"))

    assert len(list(src)) == 1


def test_eof_without_terminal_frame_ends_cleanly() -> None:
    # A stream truncated at a frame boundary (no zero-length terminal) still
    # iterates the complete frames and stops at EOF.
    frame = DataBatch(tables={"users": make_users_batch()})

    src = StreamDataSource(io.BytesIO(encode_frame(frame)))

    assert len(list(src)) == 1
