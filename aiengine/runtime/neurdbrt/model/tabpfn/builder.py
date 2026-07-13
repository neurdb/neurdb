"""TabPFN runtime model builder (in-context, streaming engine path).

TabPFN is a whole-table, in-context model: it conditions on the full labelled
context and predicts in a single forward pass rather than training with gradient
steps. It therefore plugs into the same *non-libsvm* streaming path as
``auto_pipeline`` (raw typed tokens -> DataFrame), not the dense-libsvm path:

- :meth:`train` consumes the labelled **context** batches (``stage == "train"``),
  fits the type-aware preprocessor and caches the context inside TabPFN once.
  The fitted :class:`~.stateful.StatefulTabPFN` is exposed as ``self.model`` so
  the engine ``Setup`` can stash it in :mod:`session_store` keyed by ``model_id``.
- :meth:`inference` reuses the cached context (re-loaded into ``self.model`` by
  ``Setup``) and predicts each streamed test batch, sending results back per
  batch (same protocol as ``armnet``).

The offline whole-table helper :func:`run_tabular_task` / :meth:`predict_table`
is retained for the unit test and ad-hoc use.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from neurdbrt.app.msg import InferenceResultResponse
from neurdbrt.app.ws import WebsocketSender
from neurdbrt.dataloader import StreamingDataSet
from neurdbrt.log import logger

from ..base import BuilderBase
from .model import REGRESSION, TabPFNPredictor
from .runner import run_tabular_task
from .stateful import StatefulTabPFN


class TabPFNModelBuilder(BuilderBase):
    """Builder that exposes TabPFN as an in-context runtime model."""

    def __init__(self, args):
        super().__init__(args)
        self._logger = logger.bind(model="TabPFN")
        # configured by Setup before train (BuilderBase.train signature is fixed)
        self._task_type: str = REGRESSION
        self._stype_hints: Dict[str, str] = {}
        self._id_cols: Sequence[str] = ()

    def configure(
        self,
        *,
        task_type: Optional[str] = None,
        stype_hints: Optional[Dict[str, str]] = None,
        id_cols: Optional[Sequence[str]] = None,
    ):
        if task_type is not None:
            self._task_type = task_type
        if stype_hints is not None:
            self._stype_hints = dict(stype_hints)
        if id_cols is not None:
            self._id_cols = list(id_cols)
        return self

    def _device(self) -> Optional[str]:
        # explicit override wins (handy for forcing CPU/GPU during debugging)
        return os.environ.get("NEURDB_TABPFN_DEVICE") or None

    # -- context (train) phase ------------------------------------------------

    async def train(
        self,
        train_loader: StreamingDataSet,
        val_loader: StreamingDataSet,
        test_loader: StreamingDataSet,
        epoch: int,
        train_batch_num: int,
        eva_batch_num: int,
        test_batch_num: int,
        feature_names: List[str],
        target_name: str,
    ):
        log = self._logger.bind(task="train")
        start = time.time()

        rows: List[List] = []
        labels: List = []
        batch_idx = -1
        async for batch in train_loader:
            batch_idx += 1
            values = batch["value"]
            ys = batch["y"]
            for i in range(len(values)):
                rows.append(values[i])
                labels.append(ys[i])
            if batch_idx + 1 == train_batch_num:
                break

        # drain eval/test stages so the loader/cache bookkeeping stays consistent
        await self._drain(val_loader, eva_batch_num)
        await self._drain(test_loader, test_batch_num)

        feat_df = pd.DataFrame(rows, columns=feature_names)
        log.info(
            "context collected",
            n_context=len(feat_df),
            n_features=len(feature_names),
            task_type=self._task_type,
        )

        stateful = StatefulTabPFN(
            target_col=target_name,
            task_type=self._task_type,
            feature_names=feature_names,
            stype_hints=self._stype_hints,
            id_cols=self._id_cols,
            device=self._device(),
        )
        stateful.fit_context(feat_df, labels)
        self._model = stateful

        log.info(
            "context fitted",
            n_context=stateful.n_context,
            col_to_stype=stateful.col_to_stype,
            elapsed_s=round(time.time() - start, 2),
        )

    async def _drain(self, loader: StreamingDataSet, n: int):
        if not n:
            return
        idx = -1
        async for _ in loader:
            idx += 1
            if idx + 1 == n:
                break

    # -- predict (inference) phase --------------------------------------------

    async def inference(
        self,
        data_loader: StreamingDataSet,
        inf_batch_num: int,
        feature_names: List[str],
        target_name: str,
        session_id: str,
    ):
        log = self._logger.bind(task="inference")
        stateful: StatefulTabPFN = self._model
        if not isinstance(stateful, StatefulTabPFN):
            raise RuntimeError(
                "TabPFN inference requires a fitted context; none was loaded "
                "(model_id not found in session_store)."
            )

        log.info(
            "begin inference", inf_batch_num=inf_batch_num, n_context=stateful.n_context
        )
        start = time.time()
        all_preds: List[List[float]] = []

        batch_idx = -1
        async for batch in data_loader:
            batch_idx += 1
            feat_df = pd.DataFrame(batch["value"], columns=feature_names)
            preds = stateful.predict_batch(feat_df)
            preds_list = np.asarray(preds, dtype=float).ravel().tolist()
            all_preds.append(preds_list)

            asyncio.create_task(
                WebsocketSender.send(
                    InferenceResultResponse(session_id, [preds_list]).to_json()
                )
            )
            log.info("batch predicted", batch_idx=batch_idx, n=len(preds_list))

            if batch_idx + 1 == inf_batch_num:
                break

        log.info(
            "inference end",
            batches=batch_idx + 1,
            elapsed_s=round(time.time() - start, 2),
        )
        return all_preds

    # -- offline whole-table helper (unit test / ad-hoc) ----------------------

    def predict_table(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        task_type: str = REGRESSION,
        **kwargs,
    ) -> Dict:
        return run_tabular_task(train_df, test_df, target_col, task_type, **kwargs)
