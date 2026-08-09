"""(DataBatch, DatabaseSchema) -> EncodedBatch: the composed input pipeline.

``InputPipeline`` is the one object the job layer holds. All configuration
is resolved at construction (job setup): the ``ViewBuilder`` strategy
decides how the database maps to model input — node tables, squeezed
edges, synthesized features — and publishes that decision as a static
``FeatureSchema``, which is exactly what ``FeatureConverter`` consumes. No
runtime filtering, no extra parameters. After construction the only moving
state is the ``DataBatch`` flowing through ``convert``:

    batch ──▶ view_builder.transform() ──▶ (batch', RelationGraph)
    batch' ──▶ feature_converter.apply() ──▶ EncodedFeatures
           ──▶ EncodedBatch(features, graph)

The chain reads raw -> encoded -> model-ready: ``DataBatch`` (engine wire)
-> ``EncodedBatch`` (numpy, model-agnostic — same output whatever model
the job trains) -> the model family's middleware packages it into that
family's consumable input (``model.middleware``).

Alignment guarantee: features are encoded from the transformed batch under
the same ``FeatureSchema`` the strategy derived, so ``EncodedBatch.features``
never describes a table the view compressed away, and synthesized columns
are encoded like any engine-native column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.batch import DataBatch
from data.schema import DatabaseSchema

from .feature.converter import EncodedFeatures, FeatureConverter, PipelineBuilder
from .view.base import ViewBuilder
from .view.builder import SqueezedViewBuilder
from .view.relation import RelationGraph


@dataclass(frozen=True)
class EncodedBatch:
    """The encoded form of one DataBatch: features + relational graph.

    Model-agnostic and framework-free (numpy): the same output whatever
    model the job trains. The model family's middleware packages it into
    that family's model-ready input; tabular consumers simply never read
    ``graph.edges``.
    """

    features: EncodedFeatures
    graph: RelationGraph


class InputPipeline:
    """Frozen composition of a ViewBuilder strategy and the feature path."""

    def __init__(
        self,
        schema: DatabaseSchema,
        view_builder: Optional[ViewBuilder] = None,
        pipeline_builder: Optional[PipelineBuilder] = None,
    ) -> None:
        self.schema = schema
        self.view_builder: ViewBuilder = (
            view_builder
            if view_builder is not None
            else SqueezedViewBuilder(schema=schema)
        )
        self.feature_converter = FeatureConverter(
            schema=self.view_builder.feature_schema,
            **(
                {}
                if pipeline_builder is None
                else {"pipeline_builder": pipeline_builder}
            ),
        )

    def convert(self, batch: DataBatch) -> EncodedBatch:
        batch, graph = self.view_builder.transform(batch)
        return EncodedBatch(
            features=self.feature_converter.apply(batch),
            graph=graph,
        )
