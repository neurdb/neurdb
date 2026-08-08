"""DataBatch -> RelationGraph.

``RelationConverter`` walks the schema's FK relationships and materializes a
``RelationGraph`` (node counts + typed edge index per relationship) from a
DataBatch. Independent of ``pipeline`` — both consume DataBatch directly.

Node identity is positional: node ``i`` of table ``t`` is row ``i`` of the
batch's RecordBatch for ``t``. Key values are only ever join inputs — they
never become node ids — so features (``EncodedFeatures``) and edges align by
construction because both converters read the same RecordBatch in the same
row order. Node ids are therefore batch-local; each DataBatch is a
self-contained graph.

Joins run through Arrow's native hash join (C++), which follows SQL
semantics: rows with a null in any join-key column never match. A null FK
is normal (an optional relationship) — the row stays a node, it just
contributes no edge. A null in the target (PK) columns violates a hard
RDBMS constraint the engine is supposed to enforce, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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

    Duplicate keys on either side produce one edge per matching pair (true
    inner-join semantics). On the target (PK) side keys are unique by
    construction, so this degenerates to the expected one-edge-per-FK.
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
            self._assert_target_keys_non_null(batch, rel)
            edges[rel.name] = self._join_rows(batch, rel)
        return RelationGraph(node_counts=node_counts, edges=edges)

    @staticmethod
    def _join_rows(
        batch: DataBatch, rel: RelationshipSchema
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (src_row_idx, tgt_row_idx) implementing rel as an inner join.

        Runs on Arrow's hash join: both sides are reduced to their key
        columns (renamed to a shared ``__k<i>`` namespace) plus a row-index
        column, joined in C++, and the surviving index pairs read back.
        Output is sorted by (src, tgt) so edge order is deterministic.
        """
        src_rb = batch.tables[rel.source_table]
        tgt_rb = batch.tables[rel.target_table]

        key_names = [f"__k{i}" for i in range(len(rel.source_columns))]
        src = pa.table(
            {
                **{k: src_rb.column(c) for k, c in zip(key_names, rel.source_columns)},
                "__src": np.arange(src_rb.num_rows, dtype=np.int64),
            }
        )
        tgt = pa.table(
            {
                **{k: tgt_rb.column(c) for k, c in zip(key_names, rel.target_columns)},
                "__tgt": np.arange(tgt_rb.num_rows, dtype=np.int64),
            }
        )

        joined = src.join(tgt, keys=key_names, join_type="inner").sort_by(
            [("__src", "ascending"), ("__tgt", "ascending")]
        )
        return (
            np.asarray(joined.column("__src").to_numpy(), dtype=np.int64),
            np.asarray(joined.column("__tgt").to_numpy(), dtype=np.int64),
        )

    @staticmethod
    def _assert_target_keys_non_null(
        batch: DataBatch, rel: RelationshipSchema
    ) -> None:
        """A null in the target (PK) columns is a broken hard constraint."""
        tgt_nulls = _null_key_rows(batch.tables[rel.target_table], rel.target_columns)
        if tgt_nulls:
            raise ValueError(
                f"relationship {rel.name}: {tgt_nulls} row(s) in target table "
                f"{rel.target_table} have null key column(s) "
                f"{rel.target_columns} — primary keys must be non-null; the "
                "engine sent an invalid batch"
            )


def _null_key_rows(rb: pa.RecordBatch, columns: List[str]) -> int:
    """Count rows with a null in any of the given key columns."""
    mask = None
    for c in columns:
        isnull = pc.is_null(rb.column(c))
        mask = isnull if mask is None else pc.or_(mask, isnull)
    if mask is None:
        return 0
    count = pc.sum(pc.cast(mask, pa.int64())).as_py()
    return int(count or 0)
