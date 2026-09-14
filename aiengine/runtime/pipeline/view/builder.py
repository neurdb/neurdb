"""Default ViewBuilder strategy: junction tables become edges.

``SqueezedViewBuilder`` composes the two graph mechanisms —
``RelationConverter`` (faithful schema -> graph mapping) and
``RelationSqueezer`` (junction -> edge transform) — into a strategy
satisfying the ``ViewBuilder`` contract. The batch passes through
untouched; this strategy only reshapes the graph and narrows the feature
view.

Configuration is frozen at construction (job setup):

* ``squeeze=None`` — the default rule: auto-squeeze featureless two-FK
  junction tables.
* ``squeeze=["clicks"]`` — the modeling override: squeeze the named tables
  even if they have features; those features leave the model input because
  the tables are absent from ``feature_schema``.

Other strategies (DFS-style feature synthesis, filtered subgraphs, ...)
are new classes satisfying ``ViewBuilder`` — this one is just the provided
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from data.batch import DataBatch
from data.schema import DatabaseSchema, FeatureSchema

from .relation import RelationConverter, RelationGraph
from .squeeze import RelationSqueezer


@dataclass
class SqueezedViewBuilder:
    schema: DatabaseSchema
    squeeze: Optional[Sequence[str]] = None
    _converter: RelationConverter = field(init=False, repr=False)
    _squeezer: RelationSqueezer = field(init=False, repr=False)
    _feature_schema: FeatureSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._converter = RelationConverter(schema=self.schema)
        self._squeezer = RelationSqueezer(schema=self.schema, junctions=self.squeeze)
        self._feature_schema = FeatureSchema.from_database(
            self.schema,
            tables=set(self.schema.tables) - self._squeezer.squeezed_tables,
        )

    @property
    def feature_schema(self) -> FeatureSchema:
        return self._feature_schema

    def transform(self, batch: DataBatch) -> Tuple[DataBatch, RelationGraph]:
        return batch, self._squeezer.apply(self._converter.construct(batch))
