"""Async training-job lifecycle and the transport-agnostic service surface.

The AI engine is a training-job scheduler: the DB engine submits a
``TaskDefinition`` plus its two data channels (``SchemaSource``,
``DataSource``), gets a ``Job`` handle back immediately, and polls it. The
future RPC server is a thin adapter mapping calls onto ``JobScheduler`` —
nothing in this module knows about transports.

Lifecycle::

    submit -> PENDING -> RECEIVING -> TRAINING -> SUCCEEDED
                                   \\-> FAILED / CANCELLED (from any state)

* ``RECEIVING`` — the job drains the engine's one-shot stream into a
  ``DataCache``. Data crosses the wire exactly once; everything after
  (epochs, materialization for tree/TabPFN-style models) re-iterates the
  cache locally.
* ``TRAINING`` — job setup is already frozen (task + fetched schema
  determine the ``InputPipeline`` and model construction); the training
  loop consumes the cache and reports progress through a ``MetricsSink``.
* ``SUCCEEDED`` — ``Job.result()`` returns the ``JobArtifacts``.

Interfaces only: concrete scheduler, cache, and training-loop
implementations come in later rounds, as does the model layer (a trained
model appears here only as an opaque ``bytes`` blob).
"""

from __future__ import annotations

from enum import Enum
from typing import Iterator, List, Optional, Protocol, Sequence

from data.base import NonEmptyStr, RuntimeDataModel
from data.batch import DataBatch
from data.io import DataSource, SchemaSource
from data.schema import DatabaseSchema
from pydantic import Field
from task.base import TaskDefinition


class JobStatus(str, Enum):
    PENDING = "pending"        # accepted, not yet started
    RECEIVING = "receiving"    # draining the engine's stream into the cache
    TRAINING = "training"      # loop running over the cache
    SUCCEEDED = "succeeded"    # artifacts available via Job.result()
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricRecord(RuntimeDataModel):
    """One monitored observation (e.g. step=120, name="loss", value=0.43).

    ``step`` is a monotonically increasing training-loop counter; its unit
    (batch, epoch) is the trainer's choice and irrelevant to consumers,
    which only rely on ordering.
    """

    step: int = Field(ge=0)
    name: NonEmptyStr
    value: float


class MetricsSink(Protocol):
    """Where the training loop reports progress.

    The job owns the sink; the trainer only emits. Implementations decide
    retention and exposure (in-memory history for polling, push
    notifications later).
    """

    def emit(self, record: MetricRecord) -> None: ...


class DataCache(Protocol):
    """The "transferred once" guarantee.

    ``fill`` consumes the engine's one-shot ``DataSource`` to exhaustion —
    it is called exactly once per job, during ``RECEIVING``. Afterwards the
    cache is a re-iterable sequence of the same ``DataBatch`` frames in
    arrival order, so training may take as many passes as it needs
    (epochs, full materialization) without touching the wire again.
    Storage policy (memory, spill-to-disk) is the implementation's concern.
    """

    def fill(self, source: DataSource) -> None: ...

    def __iter__(self) -> Iterator[DataBatch]: ...


class JobArtifacts(RuntimeDataModel):
    """What a finished job hands back to the DB engine.

    The preprocessing logic is not shipped as code: it is deterministically
    reconstructible from ``task`` (view/pipeline configuration) plus
    ``database_schema`` (column stypes and statistics) — the same two
    inputs that built the job's ``InputPipeline``. ``model_blob`` is
    opaque: the model layer (design deferred) owns its serialization; the
    DB engine only stores and returns it.
    """

    task: TaskDefinition
    database_schema: DatabaseSchema
    model_blob: bytes
    metrics: List[MetricRecord] = Field(default_factory=list)


class Job(Protocol):
    """Handle to one training job; all methods are safe to poll."""

    @property
    def job_id(self) -> str: ...

    @property
    def task(self) -> TaskDefinition: ...

    @property
    def status(self) -> JobStatus: ...

    def metrics(self) -> Sequence[MetricRecord]:
        """Monitoring history so far, in emission order."""
        ...

    def result(self) -> Optional[JobArtifacts]:
        """Artifacts once SUCCEEDED, else None."""
        ...

    def cancel(self) -> None:
        """Request cooperative cancellation; terminal states ignore it."""
        ...


class JobScheduler(Protocol):
    """The service surface the DB engine's RPC calls map onto.

    ``submit`` must return immediately — the job runs asynchronously and
    the caller polls the returned handle (or re-finds it via ``get`` with
    the ``task_id``-correlated ``job_id``).
    """

    def submit(
        self,
        task: TaskDefinition,
        schema_source: SchemaSource,
        data_source: DataSource,
    ) -> Job: ...

    def get(self, job_id: str) -> Optional[Job]: ...
