"""Local asynchronous training-job scheduler.

``LocalJobScheduler`` implements the ``JobScheduler`` service surface with
a bounded worker pool: ``submit`` registers a ``TrainingJob`` and returns
its handle immediately; the job runs on a worker thread through the
lifecycle pinned in ``job.base`` (PENDING -> RECEIVING -> TRAINING ->
SUCCEEDED / FAILED / CANCELLED).

The scheduler owns the two shared resources:

* **Workers** — ``max_concurrent_jobs`` bounds simultaneous training;
  excess submissions queue as PENDING.
* **Data** — a ``DatasetManager`` (see ``job.dataset``): a job pins its
  dataset while running and releases it on completion; retained datasets
  serve later tasks with the same ``dataset_key`` (RECEIVING is skipped
  entirely on a warm hit) and are LRU-evicted when new data needs room.

The TRAINING phase is delegated to an injected ``Trainer`` (the future
model layer); the scheduler never inspects batches or models.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from data.io import DataSource, SchemaSource
from task.base import TaskDefinition

from .base import (
    Job,
    JobArtifacts,
    JobStatus,
    MetricRecord,
    Trainer,
)
from .dataset import DatasetManager

_DEFAULT_DATA_BUDGET = 4 * 1024**3  # 4 GiB of cached training data

_TERMINAL = (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class _MetricsLog:
    """Thread-safe in-memory MetricsSink; the job's polling history."""

    def __init__(self) -> None:
        self._records: List[MetricRecord] = []
        self._lock = threading.Lock()

    def emit(self, record: MetricRecord) -> None:
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> Sequence[MetricRecord]:
        with self._lock:
            return tuple(self._records)


class TrainingJob:
    """Concrete ``Job``: state machine + the run executed on a worker."""

    def __init__(
        self,
        task: TaskDefinition,
        schema_source: SchemaSource,
        data_source: DataSource,
        datasets: DatasetManager,
        trainer: Trainer,
    ) -> None:
        self._task = task
        self._schema_source = schema_source
        self._data_source = data_source
        self._datasets = datasets
        self._trainer = trainer

        self._status = JobStatus.PENDING
        self._status_lock = threading.Lock()
        self._cancel = threading.Event()
        self._metrics = _MetricsLog()
        self._result: Optional[JobArtifacts] = None
        self._error: Optional[str] = None

    # -- Job protocol -------------------------------------------------------

    @property
    def job_id(self) -> str:
        return self._task.task_id

    @property
    def task(self) -> TaskDefinition:
        return self._task

    @property
    def status(self) -> JobStatus:
        with self._status_lock:
            return self._status

    @property
    def error(self) -> Optional[str]:
        return self._error

    def metrics(self) -> Sequence[MetricRecord]:
        return self._metrics.snapshot()

    def result(self) -> Optional[JobArtifacts]:
        return self._result

    def cancel(self) -> None:
        self._cancel.set()

    # -- internals ----------------------------------------------------------

    @property
    def dataset_key(self) -> str:
        """Cache identity: engine-provided dataset_key, else the task_id.

        Tasks sharing a dataset_key share cached data — the reuse contract.
        """
        key = self._task.metadata.get("dataset_key")
        return str(key) if key else self._task.task_id

    def _advance(self, status: JobStatus) -> bool:
        """Move to ``status`` unless cancelled/terminal; False stops the run."""
        with self._status_lock:
            if self._status in _TERMINAL:
                return False
            if self._cancel.is_set():
                self._status = JobStatus.CANCELLED
                return False
            self._status = status
            return True

    def _finish(self, status: JobStatus) -> None:
        with self._status_lock:
            if self._status not in _TERMINAL:
                self._status = status

    def run(self) -> None:
        """The worker-thread body. Never raises."""
        acquired = False
        try:
            if not self._advance(JobStatus.RECEIVING):
                return
            database_schema = self._schema_source.fetch()
            dataset = self._datasets.acquire(self.dataset_key, self._data_source)
            acquired = True

            if not self._advance(JobStatus.TRAINING):
                return
            model_blob = self._trainer.run(
                task=self._task,
                database_schema=database_schema,
                batches=dataset,
                metrics=self._metrics,
                should_stop=self._cancel.is_set,
            )

            if self._cancel.is_set():
                self._finish(JobStatus.CANCELLED)
                return
            self._result = JobArtifacts(
                task=self._task,
                database_schema=database_schema,
                model_blob=model_blob,
                metrics=list(self._metrics.snapshot()),
            )
            self._finish(JobStatus.SUCCEEDED)
        except Exception as exc:
            self._error = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self._finish(JobStatus.FAILED)
        finally:
            if acquired:
                self._datasets.release(self.dataset_key)


class LocalJobScheduler:
    """In-process ``JobScheduler`` over a bounded thread pool."""

    def __init__(
        self,
        trainer: Trainer,
        max_concurrent_jobs: int = 1,
        data_budget_bytes: int = _DEFAULT_DATA_BUDGET,
    ) -> None:
        self._trainer = trainer
        self._datasets = DatasetManager(budget_bytes=data_budget_bytes)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    @property
    def datasets(self) -> DatasetManager:
        return self._datasets

    def submit(
        self,
        task: TaskDefinition,
        schema_source: SchemaSource,
        data_source: DataSource,
    ) -> Job:
        job = TrainingJob(
            task=task,
            schema_source=schema_source,
            data_source=data_source,
            datasets=self._datasets,
            trainer=self._trainer,
        )
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job already submitted for task {job.job_id!r}")
            self._jobs[job.job_id] = job
        self._executor.submit(job.run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work and (optionally) wait for running jobs."""
        self._executor.shutdown(wait=wait)
