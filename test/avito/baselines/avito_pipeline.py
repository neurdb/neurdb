"""Shared pipeline core for the LOTUS / Palimpzest baselines (export-execute-import).

Faithfully mirrors the database-native workload in test/avito/workloads/,
INCLUDING its daily-rollup optimization (same algorithm, pandas instead of
SQL), so the baselines are never handicapped algorithmically:

  tool_cutoffs.sql            -> build_cutoffs()
  tool_rollups.sql            -> build_rollups()       (clicked-ads pruning +
                                 (adid, day) pre-aggregation, built once and
                                 shared across the horizon tasks)
  01_label_adctr.sql          -> build_label(h)        (from the ss rollup)
  02_features_adctr_pit.sql   -> build_features()      (from the rollups;
                                 cache-off semantics: each task computes all
                                 of its own rows)
  03_task_table.sql           -> build_task()
  08_predict_candidates.sql   -> candidate filter + TabPFN fit/predict
  09_action_list.sql          -> build_action_list()

Model + preprocessing are imported from the engine itself
(aiengine/runtime/neurdbrt/model/tabpfn: TabularPreprocessor / StatefulTabPFN),
so the baselines run EXACTLY the same type-aware preprocessing and TabPFN
configuration as NeurDB's AI operator -- only the orchestration differs.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# -- engine modules (same preprocessing / model wrapper as the AI server) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_TABPFN_DIR = os.path.normpath(
    os.path.join(_HERE, "../../../aiengine/runtime/neurdbrt/model/tabpfn")
)
if _TABPFN_DIR not in sys.path:
    sys.path.insert(0, _TABPFN_DIR)

from stateful import REGRESSION, StatefulTabPFN, pg_types_to_hints  # noqa: E402

DATA_DIR = os.path.join(_HERE, "data")
CANDIDATE_CATEGORIES = (60, 26, 27)
BATCH_SIZE = 4096
MAX_TRAIN_SAMPLES = 10_000

# Postgres types of the w_task_<h> columns (from the DB tupdesc); used to build
# the same stype hints the engine receives via the typed wire format.
TASK_PG_TYPES: Dict[str, str] = {
    "adid": "bigint",
    "ts": "timestamp without time zone",
    "split": "text",
    "price": "double precision",
    "iscontext": "double precision",
    "categoryid": "bigint",
    "locationid": "bigint",
    "title_len": "integer",
    "cat_level": "bigint",
    "cat_parent": "bigint",
    "loc_region": "double precision",
    "loc_city": "double precision",
    "ss_impr_all": "bigint",
    "ss_click_all": "double precision",
    "ss_ctr_all": "double precision",
    "ss_avgpos_all": "double precision",
    "ss_avghistctr_all": "double precision",
    "ss_impr_7d": "bigint",
    "ss_click_7d": "double precision",
    "vs_visit_all": "bigint",
    "vs_visit_7d": "bigint",
    "pr_all": "bigint",
}
TARGET_COL = "label_ctr"


# ---------------------------------------------------------------------------
# timing harness: same machine-readable lines as run_tasks.sh
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.steps: List[Tuple[str, float]] = []

    def timed(self, label: str, fn, *args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        self.steps.append((label, dt))
        print(f"TIMING,{label},{dt:.1f}", flush=True)
        return out

    def total(self) -> float:
        return sum(dt for _, dt in self.steps)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_tables(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    tables = {}
    for name in (
        "adsinfo",
        "searchstream",
        "visitstream",
        "phonerequestsstream",
        "category",
        "location",
    ):
        tables[name] = pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
    return tables


# ---------------------------------------------------------------------------
# tool_cutoffs.sql
# ---------------------------------------------------------------------------
def build_cutoffs(searchstream: pd.DataFrame) -> pd.DataFrame:
    mn = searchstream["searchdate"].min().normalize()
    mx = searchstream["searchdate"].max().normalize()
    grid = pd.date_range(mn + pd.Timedelta(days=7), mx - pd.Timedelta(days=7), freq="D")
    split = np.where(
        grid < pd.Timestamp("2015-05-11"),
        "train",
        np.where(grid < pd.Timestamp("2015-05-13"), "val", "test"),
    )
    return pd.DataFrame({"ts": grid, "split": split})


# ---------------------------------------------------------------------------
# tool_rollups.sql -- clicked-ads pruning + (adid, day) daily pre-aggregation
# ---------------------------------------------------------------------------
# Same two exactness tricks as the SQL version:
#   * day bucket = day(evt - 1us), so bucket B holds events in (B, B+1d] and
#     every half-open midnight window maps exactly onto a bucket range:
#       evt <= t          <=>  day <  t
#       evt in (t-7d, t]  <=>  day in [t-7d, t-1d]
#       evt in (t,  t+h]  <=>  day in [t,    t+h-1d]
#   * only ads with >= 1 click ever can appear in a label or feature key, so
#     the rollups are restricted to those ads up front.
# All sums/counts decompose the original aggregates exactly
# (AVG(x) = SUM(sum_x) / SUM(n_x), NULLs excluded per day like SQL AVG).
def build_rollups(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    ss = tables["searchstream"]
    clicked = ss.loc[ss["isclick"] > 0, "adid"].dropna().unique()

    def day_of(s: pd.Series) -> pd.Series:
        return (s - pd.Timedelta(microseconds=1)).dt.normalize()

    s = ss[ss["adid"].isin(clicked)]
    ss_daily = (
        s.assign(day=day_of(s["searchdate"]))
        .groupby(["adid", "day"], sort=False)
        .agg(
            impr=("searchid", "count"),
            clicks=("isclick", "sum"),
            n_click=("isclick", "count"),
            sum_pos=("position", "sum"),
            n_pos=("position", "count"),
            sum_histctr=("histctr", "sum"),
            n_histctr=("histctr", "count"),
        )
        .reset_index()
        .sort_values("day", kind="mergesort")
        .reset_index(drop=True)
    )

    v = tables["visitstream"]
    v = v[v["adid"].isin(clicked)]
    vs_daily = (
        v.assign(day=day_of(v["viewdate"]))
        .groupby(["adid", "day"], sort=False)
        .size()
        .reset_index(name="visits")
        .sort_values("day", kind="mergesort")
        .reset_index(drop=True)
    )

    p = tables["phonerequestsstream"]
    p = p[p["adid"].isin(clicked)]
    pr_daily = (
        p.assign(day=day_of(p["phonerequestdate"]))
        .groupby(["adid", "day"], sort=False)
        .size()
        .reset_index(name="reqs")
        .sort_values("day", kind="mergesort")
        .reset_index(drop=True)
    )
    return {"ss_daily": ss_daily, "vs_daily": vs_daily, "pr_daily": pr_daily}


# ---------------------------------------------------------------------------
# 01_label_adctr.sql -- label(adid, t) = SUM(isclick)/COUNT(searchid) over (t, t+h]
# ---------------------------------------------------------------------------
def build_label(
    rollups: Dict[str, pd.DataFrame], cutoffs: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    ssd = rollups["ss_daily"]
    days = ssd["day"].to_numpy()
    out = []
    for ts in cutoffs["ts"]:
        # events in (t, t+h]  <=>  day in [t, t+h-1d]
        lo = np.searchsorted(days, np.datetime64(ts), side="left")
        hi = np.searchsorted(
            days,
            np.datetime64(ts + pd.Timedelta(days=horizon_days - 1)),
            side="right",
        )
        win = ssd.iloc[lo:hi]
        g = win.groupby("adid").agg(
            clicks=("clicks", "sum"), impressions=("impr", "sum")
        )
        g = g[g["clicks"] > 0]
        g["ctr"] = g["clicks"] / g["impressions"]
        g = g.reset_index()
        g["ts"] = ts
        out.append(g[["adid", "ts", "ctr", "clicks", "impressions"]])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 02_features_adctr_pit.sql (cache-off: compute every key of this task)
# ---------------------------------------------------------------------------
def _window_aggs(
    daily: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    keys_by_ts: Dict[pd.Timestamp, np.ndarray],
    agg_fn,
) -> pd.DataFrame:
    """For each cutoff ts: aggregate rollup rows with day < ts (= events <= ts)
    and the 7d sub-window for the adids that need features at that ts.
    agg_fn(hist, win7) -> DataFrame indexed by adid."""
    days = daily["day"].to_numpy()
    out = []
    for ts in cutoffs:
        adids = keys_by_ts[ts]
        hi = np.searchsorted(days, np.datetime64(ts), side="left")
        lo7 = np.searchsorted(
            days, np.datetime64(ts - pd.Timedelta(days=7)), side="left"
        )
        hist = daily.iloc[:hi]
        hist = hist[hist["adid"].isin(adids)]
        win7 = daily.iloc[lo7:hi]
        win7 = win7[win7["adid"].isin(adids)]
        g = agg_fn(hist, win7).reindex(adids)
        g.index.name = "adid"
        g = g.reset_index()
        g["ts"] = ts
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_features(
    tables: Dict[str, pd.DataFrame],
    cutoffs: pd.DataFrame,
    label: pd.DataFrame,
    rollups: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    keys = label[["adid", "ts"]].drop_duplicates()
    cuts = sorted(keys["ts"].unique())
    keys_by_ts = {
        pd.Timestamp(ts): keys.loc[keys["ts"] == ts, "adid"].to_numpy() for ts in cuts
    }
    cuts = [pd.Timestamp(ts) for ts in cuts]

    # searchstream history aggregates (decomposed over the daily rollup)
    def ss_agg(hist: pd.DataFrame, win7: pd.DataFrame) -> pd.DataFrame:
        g = hist.groupby("adid").agg(
            impr=("impr", "sum"),
            clicks=("clicks", "sum"),
            n_click=("n_click", "sum"),
            sum_pos=("sum_pos", "sum"),
            n_pos=("n_pos", "sum"),
            sum_histctr=("sum_histctr", "sum"),
            n_histctr=("n_histctr", "sum"),
        )
        out = pd.DataFrame(index=g.index)
        out["ss_impr_all"] = g["impr"]
        out["ss_click_all"] = g["clicks"]
        out["ss_ctr_all"] = g["clicks"] / g["n_click"].replace(0, np.nan)
        out["ss_avgpos_all"] = g["sum_pos"] / g["n_pos"].replace(0, np.nan)
        out["ss_avghistctr_all"] = g["sum_histctr"] / g["n_histctr"].replace(0, np.nan)
        g7 = win7.groupby("adid").agg(
            ss_impr_7d=("impr", "sum"), ss_click_7d=("clicks", "sum")
        )
        return out.join(g7, how="outer")

    ss = _window_aggs(rollups["ss_daily"], cuts, keys_by_ts, ss_agg)
    for c, fill in (
        ("ss_impr_all", 0),
        ("ss_click_all", 0.0),
        ("ss_impr_7d", 0),
        ("ss_click_7d", 0.0),
    ):
        ss[c] = ss[c].fillna(fill)

    # visitstream
    def vs_agg(hist: pd.DataFrame, win7: pd.DataFrame) -> pd.DataFrame:
        g = hist.groupby("adid").agg(vs_visit_all=("visits", "sum"))
        g7 = win7.groupby("adid").agg(vs_visit_7d=("visits", "sum"))
        return g.join(g7, how="outer")

    vs = _window_aggs(rollups["vs_daily"], cuts, keys_by_ts, vs_agg)
    vs[["vs_visit_all", "vs_visit_7d"]] = vs[["vs_visit_all", "vs_visit_7d"]].fillna(0)

    # phone requests
    def pr_agg(hist: pd.DataFrame, win7: pd.DataFrame) -> pd.DataFrame:
        return hist.groupby("adid").agg(pr_all=("reqs", "sum"))

    pr = _window_aggs(rollups["pr_daily"], cuts, keys_by_ts, pr_agg)
    pr["pr_all"] = pr["pr_all"].fillna(0)

    # static ad attributes + dimension joins
    ads = tables["adsinfo"].copy()
    ads["title_len"] = ads["title"].str.len()
    feat = keys.merge(cutoffs, on="ts", how="left")
    feat = feat.merge(
        ads[["adid", "price", "iscontext", "categoryid", "locationid", "title_len"]],
        on="adid",
        how="left",
    )
    cat = tables["category"].rename(
        columns={"level": "cat_level", "parentcategoryid": "cat_parent"}
    )
    feat = feat.merge(
        cat[["categoryid", "cat_level", "cat_parent"]], on="categoryid", how="left"
    )
    loc = tables["location"].rename(
        columns={"regionid": "loc_region", "cityid": "loc_city"}
    )
    feat = feat.merge(
        loc[["locationid", "loc_region", "loc_city"]], on="locationid", how="left"
    )

    feat = feat.merge(ss, on=["adid", "ts"], how="left")
    feat = feat.merge(vs, on=["adid", "ts"], how="left")
    feat = feat.merge(pr, on=["adid", "ts"], how="left")

    return feat[list(TASK_PG_TYPES.keys())]


# ---------------------------------------------------------------------------
# 03_task_table.sql
# ---------------------------------------------------------------------------
def build_task(label: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    task = feat.merge(label[["adid", "ts", "ctr"]], on=["adid", "ts"], how="inner")
    return task.rename(columns={"ctr": TARGET_COL})


# ---------------------------------------------------------------------------
# 08_predict_candidates.sql -- candidate filter + in-context TabPFN
# ---------------------------------------------------------------------------
def filter_candidates(task: pd.DataFrame) -> pd.DataFrame:
    return task[task["categoryid"].isin(CANDIDATE_CATEGORIES)].reset_index(drop=True)


def predict_tabpfn(task_rows: pd.DataFrame, device: str = None) -> pd.DataFrame:
    """In-context train on the fed rows and predict the same rows -- identical
    semantics (and code) to the engine's PREDICT path with qual pushdown."""
    feat_df = task_rows.drop(columns=[TARGET_COL])
    hints = pg_types_to_hints(
        list(feat_df.columns), [TASK_PG_TYPES[c] for c in feat_df.columns]
    )

    model = StatefulTabPFN(
        target_col=TARGET_COL,
        task_type=REGRESSION,
        stype_hints=hints,
        device=device,
        max_train_samples=MAX_TRAIN_SAMPLES,
        batch_size=BATCH_SIZE,
    )
    model.fit_context(feat_df, task_rows[TARGET_COL])

    preds = []
    for i in range(0, len(feat_df), BATCH_SIZE):
        preds.append(model.predict_batch(feat_df.iloc[i : i + BATCH_SIZE]))
    nr_pred = np.concatenate(preds) if preds else np.empty(0)

    out = task_rows[["adid", "ts", "categoryid", "price", TARGET_COL]].copy()
    out["nr_pred"] = nr_pred
    return out


# ---------------------------------------------------------------------------
# 09_action_list.sql
# ---------------------------------------------------------------------------
def build_action_list(preds: Dict[int, pd.DataFrame], k: int = 10) -> pd.DataFrame:
    if not all(h in preds for h in (1, 3, 7)):
        print("(action list skipped: needs horizons 1, 3 and 7)", flush=True)
        return pd.DataFrame()
    h1, h3, h7 = preds[1], preds[3], preds[7]
    j = (
        h1.rename(columns={"nr_pred": "ctr_1d"})[
            ["adid", "ts", "categoryid", "price", "ctr_1d"]
        ]
        .merge(
            h3.rename(columns={"nr_pred": "ctr_3d"})[["adid", "ts", "ctr_3d"]],
            on=["adid", "ts"],
        )
        .merge(
            h7.rename(columns={"nr_pred": "ctr_7d"})[["adid", "ts", "ctr_7d"]],
            on=["adid", "ts"],
        )
    )

    def ntile4(s: pd.Series) -> pd.Series:
        # SQL ntile(4) over (partition by ts order by s ASC)
        n = len(s)
        order = s.rank(method="first").astype(int) - 1
        return (order * 4 // n) + 1

    for h in ("1d", "3d", "7d"):
        j[f"rk_{h}"] = j.groupby("ts")[f"ctr_{h}"].rank(method="min", ascending=False)
        j[f"q_{h}"] = j.groupby("ts")[f"ctr_{h}"].transform(ntile4)

    j["action"] = np.select(
        [
            j["rk_1d"] <= k,
            j["rk_7d"] <= k,
            (j["q_1d"] == 1) & (j["q_3d"] == 1) & (j["q_7d"] == 1),
        ],
        ["promote_now", "promote_later", "reduce_exposure"],
        default="keep",
    )
    return j


def summarize_action_list(al: pd.DataFrame) -> pd.DataFrame:
    if al.empty:
        return al
    return (
        al.groupby("action")
        .agg(
            n_ads=("adid", "count"),
            avg_ctr_1d=("ctr_1d", "mean"),
            avg_ctr_7d=("ctr_7d", "mean"),
        )
        .reset_index()
    )
