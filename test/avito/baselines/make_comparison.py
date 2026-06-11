#!/usr/bin/env python3
"""Aggregate the e2e comparison: NeurEngine settings vs LOTUS vs Palimpzest.

Reads
  test/avito/workloads/logs/results.csv        (NeurEngine 2x2 ablation)
  test/avito/baselines/logs/baseline_results.csv (this directory's runs)
Writes
  test/avito/baselines/logs/e2e_comparison.csv

One row per system with a phase breakdown (seconds):
  export | label | features | predict | other | total
"other" = load/cutoffs/cache-init/task-table/action-list glue.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NE_CSV = os.path.normpath(os.path.join(HERE, "../workloads/logs/results.csv"))
BL_CSV = os.path.join(HERE, "logs", "baseline_results.csv")
OUT = os.path.join(HERE, "logs", "e2e_comparison.csv")

rows = []

# ---- NeurEngine settings (database-native; no export) -----------------------
ne = pd.read_csv(NE_CSV).set_index("setting")
for setting, name in [
    ("cache_on__sched_on", "neurengine"),
    ("cache_off__sched_off", "neurengine_naive"),
]:
    r = ne.loc[setting]
    other = r["cutoffs_s"] + r["cache_init_s"] + r["task_s"] + r["action_list_s"]
    rows.append(
        {
            "system": name,
            "export_s": 0.0,
            "label_s": r["label_s"],
            # rollups = shared per-day stream aggregation feeding label+features;
            # folded into features to match the baselines' accounting (their
            # feature step aggregates the raw streams in pandas).
            "features_s": round(r["rollups_s"] + r["features_s"], 1),
            "predict_s": r["predict_total_s"],
            "other_s": round(other, 1),
            "total_s": r["total_s"],
        }
    )

# ---- external baselines (export-execute-import) -----------------------------
bl = pd.read_csv(BL_CSV)
for system in bl["system"].unique():
    s = bl[bl["system"] == system].set_index("step")["seconds"]
    pick = lambda prefix: float(s[s.index.str.startswith(prefix)].sum())
    export = float(s.get("export_total", 0.0))
    label = pick("01_label")
    features = pick("02_features")
    predict = pick("08_predict")
    pipeline_total = float(s.get("total", 0.0))
    other = pipeline_total - (label + features + predict)
    rows.append(
        {
            "system": system,
            "export_s": round(export, 1),
            "label_s": round(label, 1),
            "features_s": round(features, 1),
            "predict_s": round(predict, 1),
            "other_s": round(other, 1),
            "total_s": round(export + pipeline_total, 1),
        }
    )

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(f"\nwrote {OUT}")
