"""The model box: uniform containers for a heterogeneous model zoo.

A ``Model`` is a *box*, not a trainer. It unifies very different learners
(sklearn estimators, torch modules, ...) behind one exterior the rest of
the service can hold without knowing what's inside: identity, dispatch
metadata, and construction. It contains **zero training logic** — every
loop, epoch, metric emission, and cancellation check belongs to the job's
``Trainer`` (see ``model.trainer``), which selects its driving strategy
from the box's declared ``train_protocol``:

* ``FIT_ONCE``  — the library owns the loop (sklearn, TabPFN): the driver
  materializes the cached stream into arrays and calls fit exactly once.
* ``GRADIENT``  — the loop must be written by us (torch): the box declares
  *what* to compute (module, loss, optimizer); the driver owns epochs,
  batching, backward, and stepping.

``kind`` declares what shape of model input the box consumes; the
trainer validates it against the task's view before any data flows
(task defines the view -> view determines the input shape -> model is
declared against both).

Serving and serialization are deliberately absent: inference belongs to
the future infer service, and ``export()`` (compiling a trained box down
to a static artifact) is a future round.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple

from data.base import ColumnStype, MetadataKey
from data.schema import FeatureSchema


class ModelKind(str, Enum):
    """What shape of model input the box consumes."""

    SINGLE_TABLE = "single_table"  # one table's (X, y); ignores graph
    RELATIONAL = "relational"  # multi-table features + RelationGraph (GNNs)


class TrainProtocol(str, Enum):
    """How the trainer must drive the box."""

    FIT_ONCE = "fit_once"
    GRADIENT = "gradient"


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


@dataclass(frozen=True)
class CategoricalFeature:
    """A categorical input column and its embedding-table size.

    ``n_embeddings`` includes the reserved unknown code — it is
    ``len(cardinality) + 1``, matching ``CategoricalEncoder.unknown_code``.
    """

    key: Tuple[str, str]
    n_embeddings: int


@dataclass(frozen=True)
class ModelSpec:
    """Everything a box needs to build itself, frozen at job setup.

    Derived once from the view's ``FeatureSchema`` plus the task's target —
    never from data. ``feature_order`` is the canonical column order for
    matrix assembly: deterministic across fit and (future) serving.
    """

    feature_schema: FeatureSchema
    target: Tuple[str, str]
    task_type: TaskType
    n_classes: Optional[int]
    numeric_features: Tuple[Tuple[str, str], ...]
    categorical_features: Tuple[CategoricalFeature, ...]

    @property
    def feature_order(self) -> Tuple[Tuple[str, str], ...]:
        """Canonical order: numerics first, then categoricals, each sorted."""
        return self.numeric_features + tuple(f.key for f in self.categorical_features)

    @classmethod
    def derive(
        cls, feature_schema: FeatureSchema, target: Tuple[str, str]
    ) -> "ModelSpec":
        target_col = feature_schema.get_column(*target)
        if target_col is None:
            raise ValueError(f"target column {target} is not in the feature view")
        target_stats = target_col.metadata.get(MetadataKey.STATS) or {}
        target_stype = target_col.metadata.get(MetadataKey.STYPE)

        if target_stype == ColumnStype.CATEGORICAL:
            cardinality = target_stats.get(MetadataKey.CARDINALITY) or []
            if not cardinality:
                raise ValueError(f"categorical target {target} has no cardinality")
            n_classes = len(cardinality)
            task_type = TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS
        elif target_stype == ColumnStype.NUMERICAL:
            task_type, n_classes = TaskType.REGRESSION, None
        else:
            raise ValueError(
                f"target column {target} needs a numerical or categorical stype, "
                f"got {target_stype!r}"
            )

        numerics = []
        categoricals = []
        for table in sorted(feature_schema.tables):
            for col_name in sorted(feature_schema.tables[table]):
                key = (table, col_name)
                if key == tuple(target):
                    continue
                col = feature_schema.tables[table][col_name]
                stype = col.metadata.get(MetadataKey.STYPE)
                stats = col.metadata.get(MetadataKey.STATS) or {}
                if stype in (ColumnStype.NUMERICAL, ColumnStype.TIMESTAMP):
                    numerics.append(key)
                elif stype == ColumnStype.CATEGORICAL:
                    cardinality = stats.get(MetadataKey.CARDINALITY)
                    if cardinality:
                        categoricals.append(
                            CategoricalFeature(
                                key=key, n_embeddings=len(cardinality) + 1
                            )
                        )
        return cls(
            feature_schema=feature_schema,
            target=tuple(target),
            task_type=task_type,
            n_classes=n_classes,
            numeric_features=tuple(numerics),
            categorical_features=tuple(categoricals),
        )


class Model(ABC):
    """The uniform box. Subclass per family; concrete models declare the
    three class attributes and register with ``ModelDispatcher``."""

    name: ClassVar[str]
    kind: ClassVar[ModelKind]
    train_protocol: ClassVar[TrainProtocol]

    def __init__(self, spec: ModelSpec, params: Optional[Dict[str, Any]] = None):
        self.spec = spec
        self.params: Dict[str, Any] = dict(params or {})
