"""Unit test for the tabular AI-operator pipeline on local database data.

Exercises ``neurdbrt.model.tabpfn`` against an avito AdCTR task table (``w_task_<h>``)
produced by the workload SQL. Data is read from a CSV dump (preferred) or a live
DB connection.

Prepare the fixture once (from the repo root):
    bash data/workloads/dump_tasks.sh            # writes data/workloads/dump/w_task_*.csv

Run (use an env that has tabpfn, e.g. the adacontext tabpfn conda env):
    python aiengine/runtime/tests/test_tabular_pipeline.py
    # or, if pytest + tabpfn are available:
    pytest aiengine/runtime/tests/test_tabular_pipeline.py -s

Overrides via env vars:
    NEURDB_TASK_CSV   path to a w_task_<h>.csv (default: data/workloads/dump/w_task_1.csv)
    NEURDB_TABPFN_DEVICE   cpu | cuda | cuda:N   (default: auto)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the runtime package importable when run as a plain script.
RUNTIME_DIR = Path(__file__).resolve().parents[1]      # aiengine/runtime
REPO_ROOT = Path(__file__).resolve().parents[3]        # neurdb-dev
TABPFN_DIR = RUNTIME_DIR / "neurdbrt" / "model" / "tabpfn"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
# Import the TabPFN leaf modules directly (dir on sys.path) rather than through the
# `neurdbrt.model` package, whose __init__ pulls in the websocket/cache server stack
# (cache <-> app circular import) that only resolves under the live server bootstrap.
if str(TABPFN_DIR) not in sys.path:
    sys.path.insert(0, str(TABPFN_DIR))

from preprocess import TabularPreprocessor  # noqa: E402
from model import REGRESSION  # noqa: E402
from runner import run_tabular_task  # noqa: E402

try:
    import pytest
except ImportError:  # pytest not present in the tabpfn env
    pytest = None


# --- avito-specific test fixtures (kept out of the dataset-agnostic runtime) ---

# Semantic types for the avito AdCTR w_task_<h> tables, passed as hints so id-like
# integer columns are typed correctly without relying on inference.
AVITO_TASK_STYPES = {
    "adid": "drop", "ts": "timestamp", "split": "drop",
    "price": "numerical", "iscontext": "categorical",
    "categoryid": "categorical", "locationid": "categorical",
    "title_len": "numerical", "cat_level": "categorical",
    "cat_parent": "categorical", "loc_region": "categorical", "loc_city": "categorical",
    "ss_impr_all": "numerical", "ss_click_all": "numerical", "ss_ctr_all": "numerical",
    "ss_avgpos_all": "numerical", "ss_avghistctr_all": "numerical",
    "ss_impr_7d": "numerical", "ss_click_7d": "numerical",
    "vs_visit_all": "numerical", "vs_visit_7d": "numerical", "pr_all": "numerical",
    "label_ctr": "numerical",
}


def load_task_table(table, *, csv_path=None, dsn=None, split_col="split",
                    train_splits=("train", "val"), test_splits=("test",)):
    """Load a task table (CSV preferred, else live DB) and split by ``split_col``."""
    df = None
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif dsn is not None:
        import psycopg2

        conn = psycopg2.connect(**dsn)
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
        finally:
            conn.close()
    if df is None:
        raise FileNotFoundError(f"cannot load {table!r}: no csv at {csv_path!r} and no dsn")
    if split_col not in df.columns:
        raise KeyError(f"split column {split_col!r} not in {table!r}")
    train_df = df[df[split_col].isin(train_splits)].reset_index(drop=True)
    test_df = df[df[split_col].isin(test_splits)].reset_index(drop=True)
    return train_df, test_df


class _Skip(Exception):
    pass


def _skip(msg: str):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


def _fixture_csv() -> str:
    env = os.environ.get("NEURDB_TASK_CSV")
    if env:
        return env
    return str(REPO_ROOT / "data" / "workloads" / "dump" / "w_task_1.csv")


def _load():
    csv = _fixture_csv()
    dsn = None
    if not os.path.exists(csv):
        # fall back to live DB if the caller exported a DSN-friendly env
        host = os.environ.get("NEURDB_HOST")
        if host:
            dsn = {
                "host": host,
                "port": int(os.environ.get("NEURDB_PORT", "5432")),
                "dbname": os.environ.get("NEURDB_DB", "avito"),
                "user": os.environ.get("NEURDB_USER", "neurdb"),
            }
        else:
            _skip(
                f"no fixture at {csv} and NEURDB_HOST not set; "
                f"run `bash data/workloads/dump_tasks.sh` first"
            )
    table = os.environ.get("NEURDB_TASK_TABLE", "w_task_1")
    return load_task_table(table, csv_path=csv if os.path.exists(csv) else None, dsn=dsn)


def test_type_aware_preprocessing():
    """Type-aware preprocessing yields a finite, NaN-free numeric matrix."""
    train_df, test_df = _load()
    assert len(train_df) > 0 and len(test_df) > 0

    pre = TabularPreprocessor(col_to_stype=AVITO_TASK_STYPES, id_cols=["adid"])
    res = pre.fit_transform(train_df, test_df, target_col="label_ctr")

    # same feature width on train and test
    assert res.X_train.shape[1] == res.X_test.shape[1]
    assert res.X_train.shape[1] == len(res.feature_names)
    # dropped columns (adid, split) are not features
    assert "adid" not in res.feature_names
    assert "split" not in res.feature_names
    # timestamp expanded into numeric parts
    assert any(name.startswith("ts__") for name in res.feature_names)
    # categorical columns recognized
    assert "categoryid" in res.feature_names
    cat_names = {res.feature_names[i] for i in res.categorical_indices}
    assert "categoryid" in cat_names
    # numeric, finite, no NaN after imputation
    assert np.isfinite(res.X_train).all()
    assert np.isfinite(res.X_test).all()
    print(
        f"[preprocess] train={res.X_train.shape} test={res.X_test.shape} "
        f"n_features={len(res.feature_names)} n_categorical={len(res.categorical_indices)}"
    )


def test_tabpfn_regression_pipeline():
    """End-to-end: preprocess + TabPFN regression beats the mean predictor."""
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        _skip("tabpfn not installed in this environment")

    train_df, test_df = _load()
    device = os.environ.get("NEURDB_TABPFN_DEVICE")  # None -> auto
    result = run_tabular_task(
        train_df,
        test_df,
        target_col="label_ctr",
        task_type=REGRESSION,
        col_to_stype=AVITO_TASK_STYPES,
        id_cols=["adid"],
        device=device,
    )

    preds = result["predictions"]
    metrics = result["metrics"]
    assert len(preds) == result["n_test"]
    assert np.isfinite(preds).all()
    assert np.isfinite(metrics["mae"])
    # TabPFN should not be worse than predicting the training mean (small slack)
    assert metrics["mae"] <= metrics["baseline_mean_mae"] * 1.05
    print(
        f"[tabpfn] n_train={result['n_train']} n_test={result['n_test']} "
        f"n_features={result['n_features']} device={device or 'auto'}\n"
        f"         MAE={metrics['mae']:.5f} RMSE={metrics['rmse']:.5f} "
        f"baseline(mean)_MAE={metrics['baseline_mean_mae']:.5f}\n"
        f"         timings(s)={result['timings_s']}"
    )


if __name__ == "__main__":
    failures = 0
    for fn in (test_type_aware_preprocessing, test_tabpfn_regression_pipeline):
        name = fn.__name__
        try:
            fn()
            print(f"PASS {name}")
        except _Skip as e:
            print(f"SKIP {name}: {e}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
