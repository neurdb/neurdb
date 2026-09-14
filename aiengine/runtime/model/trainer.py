"""PipelineTrainer: the job's Trainer seam, wired to pipeline + model zoo.

This is where all training logic lives. The scheduler hands over
(task, schema, cached batches, metrics, should_stop); the trainer:

1. builds the view + ``InputPipeline`` from the task's ``view_config``
   (all configuration frozen before any data flows),
2. derives the ``ModelSpec`` from the view's feature schema + target,
3. dispatches the model box by ``model_name`` and checks its declared
   ``kind`` against the view: a SINGLE_TABLE box under a multi-table view
   reads only the target table — allowed, with a warning naming the
   ignored tables (their transfer was wasted, not wrong),
4. drives training with the driver matching the box's ``train_protocol``,
   each using its family's middleware:

   * FIT_ONCE  — ``SklearnMiddleware`` materializes the stream into one
     self-describing ``MatrixInput``; ``build_estimator(columns)``; one
     ``fit`` call.
   * GRADIENT  — ``TorchMiddleware`` packages each batch into a
     ``ModelInput`` (converted ONCE, cached across epochs by default —
     set ``training_config["cache_converted"] = False`` to re-convert
     streaming for datasets too large to hold twice); epoch loop:
     ``.to(device)`` -> ``module(mi)`` -> ``loss(out, mi.y)`` ->
     backward/step, per-epoch metrics, cooperative cancellation.

Returns an empty blob for now: model export/serialization is a future
round (the infer service contract).
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator

from data.batch import DataBatch
from data.schema import DatabaseSchema
from job.base import MetricRecord, MetricsSink
from pipeline.input import EncodedBatch, InputPipeline
from pipeline.view.builder import SqueezedViewBuilder
from task.base import TaskDefinition

from .base import Model, ModelKind, ModelSpec, TrainProtocol
from .dispatcher import ModelDispatcher
from .sklearn_family.middleware import SklearnMiddleware
from .torch_family.middleware import TorchMiddleware

logger = logging.getLogger(__name__)


class _EncodedStream:
    """Re-iterable view: cached DataBatches -> EncodedBatches on the fly.

    Re-iterability comes from the job's DataCache guarantee; conversion
    cost is paid per pass (the pipeline's per-column pipelines are cached,
    so repeat passes are cheap encodes, not rebuilds).
    """

    def __init__(self, pipeline: InputPipeline, batches: Iterable[DataBatch]):
        self._pipeline = pipeline
        self._batches = batches

    def __iter__(self) -> Iterator[EncodedBatch]:
        return (self._pipeline.convert(batch) for batch in self._batches)


class FitOnceDriver:
    def train(self, model, stream, training_config, metrics, should_stop) -> None:
        if should_stop():
            return
        matrix = SklearnMiddleware(model.spec)(stream)
        estimator = model.build_estimator(matrix.columns)
        estimator.fit(matrix.X, matrix.y)
        model.estimator = estimator
        metrics.emit(
            MetricRecord(
                step=1,
                name="train_score",
                value=float(estimator.score(matrix.X, matrix.y)),
            )
        )


class GradientDriver:
    def train(self, model, stream, training_config, metrics, should_stop) -> None:
        # Import is safe here: this driver only runs for a successfully
        # built GRADIENT box, which required torch to exist.
        import torch

        device = training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        module = model.build_module().to(device)
        optimizer = model.configure_optimizer(module)
        epochs = int(training_config.get("epochs", 1))
        cache_converted = bool(training_config.get("cache_converted", True))

        middleware = TorchMiddleware.for_model(model)
        converted = None
        if cache_converted:
            converted = []
            for encoded in stream:
                if should_stop():
                    return
                converted.append(middleware(encoded))

        module.train()
        step = 0
        for _epoch in range(epochs):
            if should_stop():
                break
            inputs = (
                converted
                if converted is not None
                else (middleware(encoded) for encoded in stream)
            )
            epoch_loss, n_batches = 0.0, 0
            for model_input in inputs:
                if should_stop():
                    break
                mi = model_input.to(device)
                optimizer.zero_grad()
                loss = model.loss(module(mi), mi.y)
                loss.backward()
                optimizer.step()
                step += 1
                epoch_loss += float(loss.detach())
                n_batches += 1
            metrics.emit(
                MetricRecord(
                    step=step,
                    name="epoch_loss",
                    value=epoch_loss / max(n_batches, 1),
                )
            )
        module.eval()
        model.module = module


class PipelineTrainer:
    """Implements the ``job.base.Trainer`` protocol."""

    def __init__(self) -> None:
        self._drivers = {
            TrainProtocol.FIT_ONCE: FitOnceDriver(),
            TrainProtocol.GRADIENT: GradientDriver(),
        }

    def run(
        self,
        task: TaskDefinition,
        database_schema: DatabaseSchema,
        batches: Iterable[DataBatch],
        metrics: MetricsSink,
        should_stop: Callable[[], bool],
    ) -> bytes:
        view_builder = SqueezedViewBuilder(
            schema=database_schema,
            squeeze=task.training.view_config.get("squeeze"),
        )
        pipeline = InputPipeline(schema=database_schema, view_builder=view_builder)
        spec = ModelSpec.derive(view_builder.feature_schema, tuple(task.target))
        model = ModelDispatcher.build(
            task.training.model_name, spec, task.training.model_params
        )
        self._check_kind(model, spec)

        driver = self._drivers[model.train_protocol]
        driver.train(
            model=model,
            stream=_EncodedStream(pipeline, batches),
            training_config=task.training.training_config,
            metrics=metrics,
            should_stop=should_stop,
        )
        return b""  # export/serialization: future round (infer service contract)

    @staticmethod
    def _check_kind(model: Model, spec: ModelSpec) -> None:
        if model.kind is ModelKind.SINGLE_TABLE:
            ignored = set(spec.feature_schema.tables) - {spec.target[0]}
            if ignored:
                logger.warning(
                    "model %r reads only the target table %r; view tables %s "
                    "are ignored — squeeze them or use a relational model to "
                    "consume them",
                    model.name,
                    spec.target[0],
                    sorted(ignored),
                )
