"""DataBatch -> RelationGraph.

``RelationConverter`` walks the schema's FK relationships and materializes a
``RelationGraph`` (node counts + typed edge index per relationship) from a
DataBatch. Independent of ``pipeline`` — both consume DataBatch directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
from data.batch import DataBatch
from data.schema import DatabaseSchema, RelationshipSchema


@dataclass(frozen=True)
class RelationGraph:
    """Typed relational graph derived from a DataBatch.

    ``node_counts`` is the number of rows per table (i.e., the node count for
    each node-type). ``edges`` maps ``RelationshipSchema.name`` to an
    ``(src_idx, tgt_idx)`` pair of int64 arrays indexing into the respective
    record batches.
    """

    node_counts: Dict[str, int] = field(default_factory=dict)
    edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class RelationConverter:
    """Construct a ``RelationGraph`` from a ``DataBatch``.

    Iteration is driven by the schema's relationship list. Relationships whose
    source or target table is missing from the batch are skipped — projection
    pushed down by the engine determines what's present, not the schema.
    """

    schema: DatabaseSchema

    def construct(self, batch: DataBatch) -> RelationGraph:
        node_counts: Dict[str, int] = {
            name: rb.num_rows for name, rb in batch.tables.items()
        }
        edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for rel in self.schema.relationships:
            if rel.source_table not in batch.tables:
                continue
            if rel.target_table not in batch.tables:
                continue
            edges[rel.name] = self._join_rows(batch, rel)
        return RelationGraph(node_counts=node_counts, edges=edges)

    @staticmethod
    def _join_rows(
        batch: DataBatch, rel: RelationshipSchema
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (src_row_idx, tgt_row_idx) implementing rel as an inner join."""
        src_rb = batch.tables[rel.source_table]
        tgt_rb = batch.tables[rel.target_table]

        src_keys = RelationConverter._composite_key(src_rb, rel.source_columns)
        tgt_keys = RelationConverter._composite_key(tgt_rb, rel.target_columns)

        tgt_index: Dict[Tuple[Any, ...], int] = {k: i for i, k in enumerate(tgt_keys)}

        src_idx: List[int] = []
        tgt_idx: List[int] = []
        for i, k in enumerate(src_keys):
            j = tgt_index.get(k)
            if j is None:
                continue
            src_idx.append(i)
            tgt_idx.append(j)
        return (
            np.asarray(src_idx, dtype=np.int64),
            np.asarray(tgt_idx, dtype=np.int64),
        )

    @staticmethod
    def _composite_key(rb: pa.RecordBatch, columns: List[str]) -> List[Tuple[Any, ...]]:
        cols = [rb.column(c).to_pylist() for c in columns]
        return list(zip(*cols))
