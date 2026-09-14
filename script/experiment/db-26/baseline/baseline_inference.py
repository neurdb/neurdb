#!/usr/bin/env python3
"""
Baseline inference: read data and model via SQL / DB, run inference in Python.
No WebSocket, no NeurDB pipeline — just a standalone script to compare with
  PREDICT CLASS OF ... FROM (...) TRAIN ON *;

Usage (from repo root):
  python script/experiment/db-26/baseline/baseline_inference.py
  python script/experiment/db-26/baseline/baseline_inference.py --num-batches 10

Batch size is fixed at 512. Each batch = one DB round-trip (SELECT LIMIT/OFFSET).
Tune how many batches with --num-batches (default: all).
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import time
import types
from datetime import datetime

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "api", "python"))


def _ensure_package(name: str, path: str):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [path]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_armnet_pickle_modules():
    """Load only the ARMNet model modules needed by ModelStorage.unpack()."""
    neurdbrt_dir = os.path.join(REPO_ROOT, "aiengine", "runtime", "neurdbrt")
    model_dir = os.path.join(neurdbrt_dir, "model")
    armnet_dir = os.path.join(model_dir, "armnet")

    _ensure_package("neurdbrt", neurdbrt_dir)
    _ensure_package("neurdbrt.model", model_dir)
    armnet_pkg = _ensure_package("neurdbrt.model.armnet", armnet_dir)

    _load_module("neurdbrt.model.armnet.entmax", os.path.join(armnet_dir, "entmax.py"))
    _load_module("neurdbrt.model.armnet.layer", os.path.join(armnet_dir, "layer.py"))
    armnet_model = _load_module(
        "neurdbrt.model.armnet.model", os.path.join(armnet_dir, "model.py")
    )
    armnet_pkg.ARMNetModel = armnet_model.ARMNetModel


import neurdb
import torch


def _log(level: str, msg: str, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    print(f"{ts} [{level:7}] {msg}  {extra}".strip())


def md5_list(str_list):
    return hashlib.md5(",".join(str_list).encode()).hexdigest()


def get_table_columns(conn, table_name: str):
    conn.database.cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    return [d[0] for d in conn.database.cursor.description]


def get_model_id_from_router(
    conn, table_name: str, feature_names: list, target_name: str
):
    features_hash = md5_list(feature_names)
    target_hash = md5_list([target_name])
    rows = conn.database.select(
        "router",
        ["model_id"],
        ["table_name = %s", "feature_columns = %s", "target_columns = %s"],
        [table_name, features_hash, target_hash],
    )
    if not rows:
        return None
    return int(rows[0][0])


def rows_to_batch(feature_rows, device, nfield: int):
    if not feature_rows:
        return None
    B = len(feature_rows)
    feat_id = torch.zeros((B, nfield), dtype=torch.long, device=device)
    feat_value = torch.zeros((B, nfield), dtype=torch.float, device=device)
    for i, row in enumerate(feature_rows):
        for j, v in enumerate(row):
            if j >= nfield:
                break
            feat_id[i, j] = j
            val = float(v) if v is not None else 0.0
            feat_value[i, j] = max(0.001, min(1.0, val))
    return {"id": feat_id, "value": feat_value}


BATCH_SIZE = 512


def main():
    script_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Baseline inference: SQL + load model + infer"
    )
    parser.add_argument(
        "--table", default="frappe_test", help="Table name (must match router)"
    )
    parser.add_argument("--target", default="click_rate", help="Target column name")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature column names (default: all except target)",
    )
    parser.add_argument("--db-name", default="neurdb")
    parser.add_argument("--db-user", default="neurdb")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Max number of batches to run (each batch=512 rows). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to read from table (default: all)",
    )
    parser.add_argument(
        "--out", default=None, help="Optional path to write predictions (one per line)"
    )
    args = parser.parse_args()

    conn = neurdb.NeurDB(
        db_name=args.db_name,
        db_user=args.db_user,
        db_host=args.db_host,
        db_port=args.db_port,
    )

    if args.features:
        feature_names = [s.strip() for s in args.features.split(",")]
    else:
        all_cols = get_table_columns(conn, args.table)
        if args.target not in all_cols:
            _log(
                "error",
                "target column not in table",
                table=args.table,
                target=args.target,
            )
            sys.exit(1)
        feature_names = [c for c in all_cols if c != args.target]
    nfield = len(feature_names)
    _log(
        "info",
        "using features",
        nfield=nfield,
        features_head=str(feature_names[:5]) + ("..." if nfield > 5 else ""),
    )

    model_id = get_model_id_from_router(conn, args.table, feature_names, args.target)
    if model_id is None:
        _log("error", "no model found in router", table=args.table, target=args.target)
        sys.exit(1)
    _log(
        "info",
        "found model in router",
        model_id=model_id,
        table=args.table,
        target=args.target,
    )

    _log("info", "loading model from database", model_id=model_id)
    load_start = time.perf_counter()
    try:
        _install_armnet_pickle_modules()
        storage = conn.load_model(model_id).unpack()
        model = storage.to_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception as e:
        _log("error", "model load failed", model_id=model_id, error=str(e))
        raise
    _log(
        "info",
        "model loaded successfully",
        model_id=model_id,
        device=str(device),
        load_ms=round((time.perf_counter() - load_start) * 1000),
    )

    # Per-batch query: each batch = one SELECT LIMIT/OFFSET (one DB round-trip per batch).
    cols = ", ".join(feature_names)
    limit_clause = f" LIMIT {args.limit}" if args.limit else ""
    subquery = f"SELECT {cols} FROM {args.table}{limit_clause}"
    run_batches = args.num_batches
    _log(
        "info",
        "begin inference (one query per batch, LIMIT/OFFSET)",
        batch_size=BATCH_SIZE,
        max_batches=run_batches or "all",
        task="inference",
    )
    infer_start = time.perf_counter()
    all_preds = []
    batch_index = 0
    while True:
        if run_batches is not None and batch_index >= run_batches:
            break
        if args.limit and (batch_index * BATCH_SIZE) >= args.limit:
            break
        query = f"SELECT * FROM ({subquery}) AS _ LIMIT {BATCH_SIZE} OFFSET {batch_index * BATCH_SIZE}"
        conn.database.cursor.execute(query)
        batch_rows = conn.database.cursor.fetchall()
        if not batch_rows:
            break
        batch_rows = [list(r) for r in batch_rows]
        actual_batch_size = len(batch_rows)
        batch = rows_to_batch(batch_rows, device, nfield)
        if batch is None:
            continue
        with torch.no_grad():
            y = model(batch)
            preds = y.cpu().numpy().tolist()
        all_preds.extend(preds)
        _log(
            "info",
            f"done batch for {batch_index}, Batch size: {actual_batch_size}",
            task="inference",
        )
        batch_index += 1

    infer_time = time.perf_counter() - infer_start
    _log(
        "info",
        "inference done",
        total_predictions=len(all_preds),
        batches_run=batch_index,
        infer_ms=round(infer_time * 1000),
    )

    if args.out:
        with open(args.out, "w") as f:
            for p in all_preds:
                f.write(f"{p}\n")
        _log("info", "wrote predictions to file", path=args.out)

    conn.close()
    _log(
        "info",
        "script total time",
        total_ms=round((time.perf_counter() - script_start) * 1000),
    )
    _log("info", "done")


if __name__ == "__main__":
    main()
