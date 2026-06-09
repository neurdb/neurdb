"""Local sanity check for the in-context StatefulTabPFN (engine ⑤⑥ logic).

Simulates the engine's two-phase streaming without the DB/websocket stack:

  context phase: fit_context(feat_df, y) once  -> cached "model" (by model_id)
  predict phase: predict_batch(feat_df) per streamed test batch

Run (CPU; TabPFN caps CPU context at ~1000 rows, so we subsample):

  NEURDB_TABPFN_DEVICE=cpu \
  /home/worker/miniconda3/envs/neurbench/bin/python \
  script/experiment/db-26/verify_tabpfn_stateful.py

The full-scale GPU path is validated end-to-end inside the container.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TABPFN_DIR = os.path.join(REPO, "aiengine", "runtime", "neurdbrt", "model", "tabpfn")
sys.path.insert(0, TABPFN_DIR)  # standalone (base-free) import of leaf modules

from stateful import StatefulTabPFN, infer_task_type, pg_types_to_hints  # noqa: E402

CSV = os.path.join(REPO, "test", "avito", "w_task_1.csv")
TARGET = "label_ctr"
ID_COLS = ["adid", "split"]  # entity key + experiment split flag: not features

# pg types as the engine would report them from the tupdesc (only ts matters here)
PG_TYPES = {"ts": "timestamp"}


def main():
    os.environ.setdefault("NEURDB_TABPFN_DEVICE", "cpu")
    df = pd.read_csv(CSV)
    print(f"loaded {CSV}: {df.shape[0]} rows x {df.shape[1]} cols")

    # held-out split via the table's `split` column (val == test rows)
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] != "train"].drop(columns=["split"])
    print(f"context rows={len(train_df)}  test rows={len(test_df)}")

    feature_names = [c for c in train_df.columns if c not in (TARGET,)]
    stype_hints = pg_types_to_hints(
        feature_names, [PG_TYPES.get(c, "") for c in feature_names]
    )
    print(f"stype_hints from pg types: {stype_hints}")

    # CPU TabPFN caps context at ~1000 rows -> subsample the context only.
    ctx = train_df.sample(n=min(900, len(train_df)), random_state=0)
    ctx_feat = ctx[feature_names]
    ctx_y = ctx[TARGET].tolist()

    model_id = 1
    store = {}  # stand-in for session_store

    # ---- context phase: fit once, cache by model_id ----
    stateful = StatefulTabPFN(
        target_col=TARGET,
        task_type=infer_task_type(-1),  # PREDICT VALUE -> regression
        feature_names=feature_names,
        stype_hints=stype_hints,
        id_cols=ID_COLS,
    )
    stateful.fit_context(ctx_feat, ctx_y)
    store[model_id] = stateful
    print(f"fitted context: n_context={stateful.n_context}")
    print(f"col_to_stype={stateful.col_to_stype}")

    # ---- predict phase: stream test batches against the cached context ----
    cached = store[model_id]
    batch_size = 256
    preds, ys = [], []
    test_eval = test_df.head(1024)  # keep the CPU run quick
    for i in range(0, len(test_eval), batch_size):
        chunk = test_eval.iloc[i : i + batch_size]
        p = cached.predict_batch(chunk[feature_names])
        preds.append(np.asarray(p, dtype=float))
        ys.append(chunk[TARGET].to_numpy(dtype=float))
        print(f"  predicted batch rows={len(chunk)}")

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    base = float(np.mean(np.abs(y_true - np.mean(ctx[TARGET].to_numpy(dtype=float)))))
    print(f"\nregression on {len(y_true)} test rows:")
    print(f"  TabPFN MAE={mae:.5f}  RMSE={rmse:.5f}")
    print(f"  mean-baseline MAE={base:.5f}")
    print("OK" if mae <= base + 1e-9 else "WARN: TabPFN worse than mean baseline")


if __name__ == "__main__":
    main()
