"""Model-layer view of the input contract.

``ModelInput`` is produced by ``pipeline.input.InputPipeline`` — the
pipeline is the composition layer that turns (DataBatch, DatabaseSchema)
into model input, so the type lives with the producer. Re-exported here so
model code keeps importing it from the layer it belongs to conceptually.
"""

from pipeline.input import ModelInput

__all__ = ["ModelInput"]
