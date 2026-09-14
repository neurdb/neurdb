"""The shared task contract between the DB engine and the AI engine.

A *task* is what the DB engine (Postgres side) submits over RPC to request
work from the AI engine. ``TaskDefinition`` is the wire contract both sides
agree on: it must stay serializable (``to_dict`` / ``from_dict`` on
``RuntimeDataModel``) and evolve compatibly — the DB engine is an external
system, not a co-deployed module.

Division of labor (the DB engine side is future work; only the interface is
fixed here):

* The **DB engine** owns data preparation: the user defines the task in
  SQL, and the engine splits/samples the training data, computes global
  column statistics into ``ColumnSchema.metadata``, and streams the result.
* The **AI engine** receives this ``TaskDefinition`` together with a
  ``SchemaSource`` (one ``DatabaseSchema`` fetch) and a ``DataSource``
  (one-shot ``DataBatch`` stream) — both defined in ``data.io`` — and runs
  the training job asynchronously (see ``job.base``).

Everything in a task is frozen job configuration: task + fetched schema
fully determine the input pipeline (``pipeline.InputPipeline``) and model
construction before the first batch flows. Inference is out of scope — it
belongs to a future infer engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple

from data.base import NonEmptyStr, RuntimeDataModel
from pydantic import ConfigDict, Field


class TaskKind(str, Enum):
    """What the engine asks for. Only training exists today; future kinds
    (fine-tuning, evaluation, ...) extend this enum — the field is part of
    the wire contract, so values must never be reused or renamed."""

    TRAIN = "train"


class TrainingSpec(RuntimeDataModel):
    """How to train: everything the AI engine needs to configure a job.

    ``model_name`` keys into the model registry (design deferred; e.g.
    ``"logreg"``, ``"deepfm"``). The three config dicts are free-form on
    the wire and interpreted by their consumers:

    * ``model_params``    — hyperparameters for the model factory.
      (Named ``model_params`` because ``model_config`` is reserved by
      pydantic.)
    * ``view_config``     — ViewBuilder strategy configuration
                            (e.g. ``{"squeeze": ["clicks"]}``).
    * ``training_config`` — loop budget: epochs/steps, early stopping, ...
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: NonEmptyStr
    model_params: Dict[str, Any] = Field(default_factory=dict)
    view_config: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(RuntimeDataModel):
    """One unit of work the DB engine submits.

    ``task_id`` is assigned by the DB engine and is the correlation key for
    everything that follows (job lookup, metrics polling, artifact
    retrieval). ``target`` names the label column as ``(table, column)`` —
    it must carry an ``stype`` in the streamed schema so the pipeline can
    encode it.
    """

    task_id: NonEmptyStr
    kind: TaskKind = TaskKind.TRAIN
    target: Tuple[NonEmptyStr, NonEmptyStr]
    training: TrainingSpec
    metadata: Dict[str, Any] = Field(default_factory=dict)
