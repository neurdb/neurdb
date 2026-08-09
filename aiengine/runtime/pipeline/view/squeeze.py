"""RelationGraph -> RelationGraph: squeeze featureless junction tables.

``RelationSqueezer`` is the default graph-modeling rule: a *junction* table
(two FKs, nothing to embed) carries no semantics of its own — it exists only
to encode a many-to-many relationship — so instead of a featureless node
type with two edge types, it becomes one direct edge type between the two
entity tables:

    before:  orders <-- orders_products --> products
    after:   orders <----------------------> products   (edge "orders_products")

This is a policy, deliberately separate from ``RelationConverter`` (which
stays a faithful schema -> graph mapping). Other graph-modeling strategies
can be implemented as further ``RelationGraph -> RelationGraph`` transforms
and swapped in by the task layer.

Feature alignment is automatic: a table qualifies for squeezing only if it
has no FEATURE/TARGET/stype-bearing columns, and exactly those columns are
what ``FeatureConverter`` encodes — so a squeezed table never had features
to begin with, and surviving tables keep their row order (node ids) intact.

The squeeze *plan* is derived once from the ``DatabaseSchema`` (it is
schema-static); applying it per batch is a cheap numpy index-join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from data.base import ColumnRole
from data.schema import DatabaseSchema, RelationshipSchema

from .relation import RelationGraph

# Roles a junction table may contain: keys and free-form metadata only.
# Anything else (FEATURE, TARGET, TIMESTAMP) means the table has its own
# semantics and must stay a node type.
_KEY_ROLES = (ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.METADATA)


@dataclass(frozen=True)
class _SqueezePlanEntry:
    """One squeezable junction: its two outgoing FK relationships.

    The composed edge runs ``rel_a.target_table -> rel_b.target_table``,
    ordered by the relationships' position in ``schema.relationships``, and
    is named after the junction table.
    """

    junction: str
    rel_a: RelationshipSchema
    rel_b: RelationshipSchema


@dataclass
class RelationSqueezer:
    """Squeeze junction tables into direct edges.

    Two modes, chosen at construction (schema-static either way):

    * ``junctions=None`` (default rule) — auto-detect: a table is squeezable
      iff every column's role is PRIMARY_KEY / FOREIGN_KEY / METADATA
      (nothing the feature pipeline would encode), it is the source of
      exactly two FK relationships, and no relationship targets it.
    * ``junctions=[...]`` (strategy override) — squeeze exactly the named
      tables, even if they carry feature columns (e.g. a ``clicks`` table
      modeled as user->item edges). Structural requirements still hold —
      two outgoing FK relationships, not targeted — and are validated
      eagerly, raising ``ValueError`` on violation. The caller owns the
      consequence that the table's features leave the model input; the
      composition layer keeps the feature path aligned via
      ``GraphBuilder.node_tables``.

    The two FK ends may be the same table (a self-relation like
    ``friendships``). At apply time a junction is squeezed only when both
    of its edge lists made it into the graph (both entity tables present in
    the batch); otherwise the junction passes through unchanged.
    """

    schema: DatabaseSchema
    junctions: Optional[Sequence[str]] = None
    _plan: List[_SqueezePlanEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._plan = self._build_plan(self.schema, self.junctions)

    @property
    def squeezed_tables(self) -> FrozenSet[str]:
        """Tables this squeezer removes as node types (schema-static)."""
        return frozenset(entry.junction for entry in self._plan)

    @staticmethod
    def _build_plan(
        schema: DatabaseSchema, junctions: Optional[Sequence[str]]
    ) -> List[_SqueezePlanEntry]:
        outgoing: Dict[str, List[RelationshipSchema]] = {}
        targeted = set()
        for rel in schema.relationships:
            outgoing.setdefault(rel.source_table, []).append(rel)
            targeted.add(rel.target_table)

        plan: List[_SqueezePlanEntry] = []

        if junctions is not None:
            for name in junctions:
                if name not in schema.tables:
                    raise ValueError(f"cannot squeeze {name}: not in schema")
                rels = outgoing.get(name, [])
                if len(rels) != 2:
                    raise ValueError(
                        f"cannot squeeze {name}: needs exactly two outgoing FK "
                        f"relationships, found {len(rels)}"
                    )
                if name in targeted:
                    raise ValueError(
                        f"cannot squeeze {name}: other relationships target it, "
                        "their edges would dangle"
                    )
                plan.append(
                    _SqueezePlanEntry(junction=name, rel_a=rels[0], rel_b=rels[1])
                )
            return plan

        for name, table in schema.tables.items():
            rels = outgoing.get(name, [])
            if len(rels) != 2 or name in targeted:
                continue
            if any(col.role not in _KEY_ROLES for col in table.columns.values()):
                continue
            plan.append(_SqueezePlanEntry(junction=name, rel_a=rels[0], rel_b=rels[1]))
        return plan

    def apply(self, graph: RelationGraph) -> RelationGraph:
        if not self._plan:
            return graph

        node_counts = dict(graph.node_counts)
        edges = dict(graph.edges)

        for entry in self._plan:
            n_junction = node_counts.get(entry.junction)
            if n_junction is None:
                continue  # junction not in this batch
            edge_a = edges.get(entry.rel_a.name)
            edge_b = edges.get(entry.rel_b.name)
            if edge_a is None or edge_b is None:
                continue  # an entity side is missing; leave the junction alone

            edges[entry.junction] = self._compose(n_junction, edge_a, edge_b)
            del edges[entry.rel_a.name]
            del edges[entry.rel_b.name]
            del node_counts[entry.junction]

        return RelationGraph(node_counts=node_counts, edges=edges)

    @staticmethod
    def _compose(
        n_junction: int,
        edge_a: Tuple[np.ndarray, np.ndarray],
        edge_b: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Join the two FK edge lists on the junction row index.

        Each junction row holds one FK value per side (the target is a PK,
        unique), so it maps to at most one row of each entity table. Rows
        with a null FK on either side are absent from that edge list and
        produce no composed edge.
        """
        a_of = np.full(n_junction, -1, dtype=np.int64)
        b_of = np.full(n_junction, -1, dtype=np.int64)
        a_of[edge_a[0]] = edge_a[1]
        b_of[edge_b[0]] = edge_b[1]
        mask = (a_of >= 0) & (b_of >= 0)
        return a_of[mask], b_of[mask]
