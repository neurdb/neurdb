import json
import struct
from typing import Any, Dict, Optional

import pyarrow as pa
from pydantic import Field, model_validator

from .base import ArrowRuntimeModel


_MAGIC = b"NEURDBDB"
_HEADER_LEN_FMT = "<I"
_HEADER_LEN_SIZE = struct.calcsize(_HEADER_LEN_FMT)


def _record_batch_to_ipc(rb: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, rb.schema) as writer:
        writer.write_batch(rb)
    return sink.getvalue().to_pybytes()


def _record_batch_from_ipc(payload: bytes) -> pa.RecordBatch:
    with pa.ipc.open_stream(pa.BufferReader(payload)) as reader:
        return next(iter(reader))


class DataBatch(ArrowRuntimeModel):
    tables: Dict[str, pa.RecordBatch]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_batch(self) -> "DataBatch":
        if not self.tables:
            raise ValueError("data batch must contain at least one table")
            
        return self

    def get_column(self, table: str, column: str) -> Optional[pa.Array]:
        rb = self.tables.get(table)
        if rb is None or column not in rb.schema.names:
            return None
        return rb.column(column)

    def to_bytes(self) -> bytes:
        table_payloads: Dict[str, bytes] = {
            name: _record_batch_to_ipc(rb) for name, rb in self.tables.items()
        }

        header = {
            "metadata": self.metadata,
            "tables": [
                {"name": name, "length": len(payload)}
                for name, payload in table_payloads.items()
            ],
        }
        header_bytes = json.dumps(header).encode("utf-8")

        buffer = bytearray()
        buffer += _MAGIC
        buffer += struct.pack(_HEADER_LEN_FMT, len(header_bytes))
        buffer += header_bytes
        for entry in header["tables"]:
            buffer += table_payloads[entry["name"]]
        return bytes(buffer)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DataBatch":
        if not data.startswith(_MAGIC):
            raise ValueError("invalid DataBatch payload: magic header missing")

        offset = len(_MAGIC)
        (header_len,) = struct.unpack(
            _HEADER_LEN_FMT, data[offset : offset + _HEADER_LEN_SIZE]
        )
        offset += _HEADER_LEN_SIZE
        header = json.loads(data[offset : offset + header_len])
        offset += header_len

        tables: Dict[str, pa.RecordBatch] = {}
        for entry in header["tables"]:
            length = entry["length"]
            payload = data[offset : offset + length]
            offset += length
            tables[entry["name"]] = _record_batch_from_ipc(payload)

        return cls(
            tables=tables,
            metadata=header.get("metadata") or {},
        )
