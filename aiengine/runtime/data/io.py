"""IO surface between the DB engine and the AI runtime.

Two decoupled protocols, because schema and data are genuinely different
things with different lifecycles:

* :class:`SchemaSource` — pull the :class:`DatabaseSchema` once
  (request/response; small, stable metadata).
* :class:`DataSource` — iterate :class:`DataBatch` frames (a stream; large
  Arrow payloads, ephemeral per run).

The DB engine is the producer and the driver. Each ``DataBatch`` frame is an
entity-anchored relational unit the engine has already assembled and validated
by construction (it followed real FK edges and did the sampling). The runtime
therefore trusts every frame as-is — no schema-consistency check, no sampling
on this side.

A one-shot delivery is simply a ``DataSource`` that yields a single frame; it
is not a separate protocol. The caller (the task/job layer) binds the one
fetched schema to each streamed batch.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Iterable, Iterator, Protocol

from .batch import DataBatch
from .schema import DatabaseSchema

# Frame = 4-byte little-endian payload length + DataBatch IPC bytes.
# A zero-length frame (or EOF) terminates the stream.
_LEN_FMT = "<I"
_LEN_SIZE = struct.calcsize(_LEN_FMT)


class SchemaSource(Protocol):
    """Pull the DatabaseSchema once — request/response, metadata not data."""

    def fetch(self) -> DatabaseSchema: ...


class DataSource(Protocol):
    """Stream entity-anchored DataBatch frames; trust each frame as-is.

    A one-shot delivery is a stream of length one.
    """

    def __iter__(self) -> Iterator[DataBatch]: ...


# ---------------------------------------------------------------------------
# Framed-stream transport
# ---------------------------------------------------------------------------


def encode_frame(batch: DataBatch) -> bytes:
    """Serialize one DataBatch as a length-prefixed frame."""
    payload = batch.to_bytes()
    return struct.pack(_LEN_FMT, len(payload)) + payload


def encode_stream(batches: Iterable[DataBatch]) -> bytes:
    """Serialize a sequence of DataBatches into a frame stream with a
    zero-length terminal frame. Mirror of :class:`StreamDataSource`."""
    out = bytearray()
    for batch in batches:
        out += encode_frame(batch)
    out += struct.pack(_LEN_FMT, 0)  # terminal
    return bytes(out)


class StreamDataSource:
    """Read length-prefixed DataBatch frames off a binary stream.

    Works over anything with ``read(n)`` — ``io.BytesIO`` in tests,
    ``socket.makefile("rb")`` in production. Satisfies :class:`DataSource`.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[DataBatch]:
        while True:
            header = self._readn(_LEN_SIZE)
            if len(header) < _LEN_SIZE:
                return  # EOF
            (length,) = struct.unpack(_LEN_FMT, header)
            if length == 0:
                return  # terminal frame
            yield DataBatch.from_bytes(self._readn(length))

    def _readn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._stream.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)
