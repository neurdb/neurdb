"""ViewBuilder: the strategy that builds the model's view of the database.

A ``ViewBuilder`` decides how the relational database maps into model
input — which tables are node types, which collapse into edges, and which
compress into synthesized features (e.g. Deep Feature Synthesis aggregates).
It is constructed once per job from the static ``DatabaseSchema`` plus
strategy configuration; per batch it is a pure function of the data.

Its two outputs are the unified contract of the stage:

* ``feature_schema`` — schema-static, derived at construction: the slim
  ``FeatureSchema`` view of exactly what the feature path should encode
  (surviving tables' columns plus any synthesized columns). This is the
  alignment contract the composition layer hands to ``FeatureConverter``.
* ``transform(batch)`` — per batch: the (possibly rewritten) ``DataBatch``
  — synthesized columns added, compressed tables dropped — plus the
  ``RelationGraph``.

``DatabaseSchema`` itself is never touched: it stays the engine's static
ground truth; every derived decision lives in the builder and its
``FeatureSchema``.
"""

from __future__ import annotations

from typing import Protocol, Tuple

from data.batch import DataBatch
from data.schema import FeatureSchema

from .relation import RelationGraph


class ViewBuilder(Protocol):
    @property
    def feature_schema(self) -> FeatureSchema:
        """What the feature path encodes. Schema-static: the same answer
        for every batch."""
        ...

    def transform(self, batch: DataBatch) -> Tuple[DataBatch, RelationGraph]:
        """View of one batch: (possibly rewritten batch, graph). Rows of
        surviving tables keep their positional node ids."""
        ...
