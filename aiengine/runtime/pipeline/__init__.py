"""The input pipeline: (DataBatch, DatabaseSchema) -> ModelInput.

Two stages composed by ``InputPipeline``:

* ``pipeline.view`` — the ViewBuilder strategy: how the relational database
  maps to model input (node tables, squeezed edges, synthesized features).
* ``pipeline.feature`` — column encoding: schema-driven per-column
  pipelines producing framework-agnostic numpy.

The public surface for the job layer is ``InputPipeline``; everything else
is exported for strategy authors and tests.
"""

from .feature import (
    EncodedFeatures,
    EncodeError,
    FeatureConverter,
    PipelineBuilder,
)
from .input import InputPipeline, ModelInput
from .view import (
    RelationConverter,
    RelationGraph,
    RelationSqueezer,
    SqueezedViewBuilder,
    ViewBuilder,
)

__all__ = [
    "InputPipeline",
    "ModelInput",
    "EncodedFeatures",
    "EncodeError",
    "FeatureConverter",
    "PipelineBuilder",
    "ViewBuilder",
    "SqueezedViewBuilder",
    "RelationConverter",
    "RelationGraph",
    "RelationSqueezer",
]
