"""Composition contract for the model layer.

``ModelInput`` is a single immutable struct the model layer consumes: pure
composition of the two parallel converter outputs, zero logic. The model is the
consumer, so the contract lives with the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.relation import RelationGraph
from pipeline.converter import EncodedFeatures


@dataclass(frozen=True)
class ModelInput:
    """Composed model input: encoded features + relational graph.

    Tabular models simply ignore ``graph.edges``.
    """

    features: EncodedFeatures
    graph: RelationGraph
