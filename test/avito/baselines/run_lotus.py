#!/usr/bin/env python3
"""LOTUS baseline for the avito AdCTR horizon-sweep workload (export-execute-import).

LOTUS (lotus-ai) is a semantic-operator framework over pandas DataFrames. Its
sem_* operators are LLM-backed and have no tabular-prediction operator, so the
idiomatic LOTUS implementation of this workload is a pandas pipeline (LOTUS's
own dataframe substrate) with TabPFN invoked as opaque user code -- exactly the
export-execute-import pattern: no optimizer sees the relational work and the
model call together, and nothing is shared across the horizon tasks.

The daily rollups (same algorithmic optimization as the engine's
tool_rollups.sql) are built once and shared across the horizon tasks; each
horizon task then runs label -> PIT features -> task -> candidates -> TabPFN
on top of them, and finally the cross-horizon action list.

Run (in the bl_lotus conda env):
    python run_lotus.py [--device cuda:0] [--horizons 1 3 7] [--k 10]
"""

from __future__ import annotations

import argparse
import os
import time

import avito_pipeline as core
import lotus  # noqa: F401  -- framework under test (sem ops unused: LLM-only)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data", default=core.DATA_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "lotus"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = core.Timer()
    t_start = time.time()

    tables = t.timed("load_tables", core.load_tables, args.data)
    cutoffs = t.timed("tool_cutoffs", core.build_cutoffs, tables["searchstream"])
    rollups = t.timed("tool_rollups", core.build_rollups, tables)

    preds = {}
    for h in args.horizons:
        label = t.timed("01_label_h%d" % h, core.build_label, rollups, cutoffs, h)
        feat = t.timed(
            "02_features_h%d" % h, core.build_features, tables, cutoffs, label, rollups
        )
        task = t.timed("03_task_h%d" % h, core.build_task, label, feat)

        def predict_step(task_df):
            cand = core.filter_candidates(task_df)
            return core.predict_tabpfn(cand, device=args.device)

        pred = t.timed("08_predict_h%d" % h, predict_step, task)
        rmse = float(np.sqrt(np.mean((pred["nr_pred"] - pred[core.TARGET_COL]) ** 2)))
        print(
            f"  h={h}: task_rows={len(task)} candidates={len(pred)} rmse={rmse:.4f}",
            flush=True,
        )
        pred.to_parquet(os.path.join(args.out, f"w_pred_{h}.parquet"), index=False)
        preds[h] = pred

    action = t.timed("09_action_list", core.build_action_list, preds, args.k)
    action.to_csv(os.path.join(args.out, "action_list.csv"), index=False)
    print(core.summarize_action_list(action).to_string(index=False), flush=True)

    print(f"TIMING,total,{time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
