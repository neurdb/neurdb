"""TabPFN runtime model builder.

Plugs TabPFN into the ``neurdbrt.model`` builder pattern (like ``armnet``).
The actual whole-table logic lives in the base-free :mod:`runner` so it can be
unit-tested without dragging in the websocket/cache server stack.

TabPFN is an in-context (whole-table) model: it conditions on the full training
set at inference time rather than training with gradient steps over libsvm
batches. The streaming ``train`` / ``inference`` hooks of ``BuilderBase`` are
therefore left as explicit TODOs to be wired into the engine's whole-table path
during e2e, while ``predict_table`` / ``run_tabular_task`` provide the working
path used today.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..base import BuilderBase
from .model import REGRESSION, TabPFNPredictor
from .runner import run_tabular_task


class TabPFNModelBuilder(BuilderBase):
    """Builder that exposes TabPFN as a runtime model."""

    def __init__(self, args):
        super().__init__(args)
        self._predictor: Optional[TabPFNPredictor] = None

    def predict_table(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        task_type: str = REGRESSION,
        **kwargs,
    ) -> Dict:
        """Convenience whole-table wrapper over :func:`run_tabular_task`."""
        return run_tabular_task(train_df, test_df, target_col, task_type, **kwargs)

    async def train(self, *args, **kwargs):
        raise NotImplementedError(
            "TabPFN is in-context (no gradient training). Wire the whole-table path "
            "into the engine during e2e; use predict_table()/run_tabular_task() meanwhile."
        )

    async def inference(self, *args, **kwargs):
        raise NotImplementedError(
            "TabPFN inference needs the assembled feature table, not libsvm batches. "
            "Wire the whole-table path into the engine during e2e; use "
            "predict_table()/run_tabular_task() meanwhile."
        )
