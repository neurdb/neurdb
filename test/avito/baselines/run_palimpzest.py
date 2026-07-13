#!/usr/bin/env python3
"""Palimpzest baseline for the avito AdCTR horizon-sweep workload.

Palimpzest (PZ) is a declarative AI-analytics system: Dataset pipelines of
filter / map(UDF) / sem_* operators compiled by a cost-based optimizer. It has
no tabular-prediction operator and its UDF interface is strictly one record at
a time (NonLLMConvert calls ``udf(record_dict)``), so an in-context model like
TabPFN cannot be expressed as a batch operator. The idiomatic implementation:

  * relational prep (labels, PIT features, task table) happens OUTSIDE PZ in
    pandas (PZ has no temporal/PIT primitives) -- shared avito_pipeline core,
    identical preprocessing to NeurEngine's AI operator;
  * the per-task prediction query IS a PZ pipeline:
        MemoryDataset(w_task_h) -> filter(candidates) -> map(predict_ctr)
    run with the MinCost policy. The UDF fits the TabPFN context lazily on
    the candidate set once, batch-predicts, memoizes, and serves PZ's
    per-record calls from the memo (the most favorable integration possible).

The daily rollups (same algorithmic optimization as the engine's
tool_rollups.sql) are built once in pandas and shared across the horizon
tasks; each per-horizon prediction query is an independent PZ pipeline run
(PZ itself has no cross-run materialization/reuse).

Run (in the bl_pz conda env):
    python run_palimpzest.py [--device cuda:0] [--horizons 1 3 7] [--k 10]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

# PZ validates LLM API keys at processor startup even for pure-UDF plans.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-not-used-pure-udf-plan")

import avito_pipeline as core
import palimpzest as pz

HERE = os.path.dirname(os.path.abspath(__file__))


def make_predict_udf(candidates: pd.DataFrame, device: str):
    """Per-record UDF over PZ's one-record-at-a-time interface.

    Fits the in-context TabPFN ONCE on the candidate rows (the rows this
    operator will be called on, given the upstream candidate filter), batch
    predicts, and memoizes -- so PZ still gets batched GPU inference despite
    its record-level UDF contract."""
    memo: dict = {}

    def predict_ctr(record: dict) -> dict:
        if not memo:
            pred = core.predict_tabpfn(candidates, device=device)
            for adid, ts, p in zip(pred["adid"], pred["ts"], pred["nr_pred"]):
                memo[(int(adid), str(pd.Timestamp(ts)))] = float(p)
        key = (int(record["adid"]), str(pd.Timestamp(record["ts"])))
        return {"nr_pred": memo[key]}

    return predict_ctr


def run_pz_predict(task: pd.DataFrame, device: str) -> pd.DataFrame:
    candidates = core.filter_candidates(task)
    udf = make_predict_udf(candidates, device)

    ds = pz.MemoryDataset(id="w_task", vals=task)
    plan = ds.filter(
        lambda r: r["categoryid"] in core.CANDIDATE_CATEGORIES,
        depends_on=["categoryid"],
    )
    plan = plan.map(
        udf=udf,
        cols=[
            {
                "name": "nr_pred",
                "type": float,
                "description": "Predicted CTR of the ad over the task horizon",
            }
        ],
    )

    config = pz.QueryProcessorConfig(
        policy=pz.MinCost(),
        execution_strategy="sequential",
        progress=False,
    )
    out = plan.run(config)
    df = out.to_df()

    df = df[["adid", "ts", "categoryid", "price", core.TARGET_COL, "nr_pred"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data", default=core.DATA_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "palimpzest"))
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

        pred = t.timed("08_predict_h%d" % h, run_pz_predict, task, args.device)
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
