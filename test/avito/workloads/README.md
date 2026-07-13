# Avito AdCTR horizon-sweep workload (database-native)

SQL that builds the end-to-end workload tables for the revision experiment,
directly inside NeurDB (database `avito`). Replaces offline Featuretools DFS with
in-database point-in-time (PIT) joins, and demonstrates **cross-task feature
reuse via a PIT feature cache**.

## Two kinds of SQL

- **LABEL** (`01_label_adctr.sql`): future window `(t, t+h]`, horizon-dependent.
- **FEATURE / PIT** (`02_features_adctr_pit.sql`): history `(.., t]`,
  horizon-independent -> a given `(adid, t)` has the same features for any task,
  so it can be cached and reused across horizons.

## Tables (in db `avito`, schema `public`)

| Table | Built | Meaning |
|-------|-------|---------|
| `w_cutoffs` | once | prediction-time grid + train/val/test split |
| `w_ad_label_<h>` | per task | CTR target over `(t, t+h]` (clicked ads) |
| `w_feat_cache` | grows | cached PIT features keyed by `(adid, t)` |
| `w_task_<h>` | per task | flat table = `w_ad_label_<h>` (x) `w_feat_cache` (features + `label_ctr`) |
| `w_pred_<h>` | per task | candidate rows + `nr_pred` (CTAS over nested `PREDICT`, by `08`) |
| `w_action_list` | once | final cross-horizon action list (`09`) |

`w_task_<h>` is the supervised table fed to the AI operator (TabPFN).

## Reuse vs no-reuse = one knob: when to reset the cache

Both modes run the **same** per-task feature SQL (`02_features`). The only
difference is how often the cache (`tool_feat_cache_init`) is reset:

- **REUSE (cache ON):** reset cache **once**; each task computes only the
  `(adid, t)` rows not already cached (its *delta*). Total feature rows computed
  across tasks = the union (no key computed twice).
- **NO-REUSE (cache OFF):** reset cache **before every task**; each task
  recomputes all of its own rows from scratch (overlapping keys recomputed).

So for horizons 1/3/7 (nested windows, 1d-keys ⊂ 3d-keys ⊂ 7d-keys):

| | rows computed |
|---|---|
| reuse | 8368 + 10738 + 14425 = **33531** (union, each key once) |
| no-reuse | 8368 + 19106 + 33531 = **60005** (overlap recomputed) |

This is exactly the optimizer/runtime co-design point: a reuse-aware runtime keeps
the PIT feature artifact and skips already-computed keys; a naive system reruns
the whole feature computation per task.

## The prediction + aggregation tasks (AI operator inside SQL)

- **`08_predict_candidates.sql`** (per horizon): candidate-scoped prediction in
  one statement -- `CREATE TABLE w_pred_<h> AS SELECT ... FROM (PREDICT ...) p
  WHERE <cand>`. The `:sched` knob toggles AI-operator scheduling:
  `off` = operator pinned at root, candidate filter evaluated above it (all
  rows trained/inferred); `on` = `nr_predict_pushdown` lets the planner sink
  the input-column filter below the operator (candidates only). Prints the
  `EXPLAIN` so the chosen schedule + `cost_neurdbpredict` pricing is visible.
- **`09_action_list.sql`**: joins `w_pred_1/3/7` on `(adid, ts)` and maps each
  candidate's prediction profile to `promote_now` / `promote_later` / `keep` /
  `reduce_exposure` (per-cutoff top-k and quartile rules).

Demos kept for reference: `04` (top-level PREDICT), `05` (predict_into
materialization), `06` (nested PREDICT forms), `07` (pushdown A/B on one query).

## Run

```bash
bash run_tasks.sh     # FULL task set: prep + 08 per horizon (off+on) + 09
bash run_matrix.sh    # 2x2 ablation: cache reuse x AI-op scheduling -> logs/ + results.csv
bash run_avito.sh     # prep only (reuse mode): builds w_task_1/3/7
bash bench.sh         # times reuse vs no-reuse feature compute + rows computed
```

`run_tasks.sh` knobs (env): `HORIZONS="1 3 7"`, `CAND="categoryid IN (60, 26, 27)"`,
`K=10` (action-list top-k), `MODES="off on"` (AI-operator scheduling),
`CACHE=on|off` (reset feature cache once vs before every task), `SKIP_PREP=1`
(reuse existing `w_task_<h>`). Each step emits a `TIMING,<step>,<seconds>` line
plus a final summary; reference run (cache on): baseline vs dynamic predict =
12.9/1.6s (h=1), 24.7/2.7s (h=3), 39.6/5.3s (h=7).

`run_matrix.sh` runs the 4 settings (cache on/off x sched on/off), keeps each
full log in `logs/<setting>.log`, and aggregates the TIMING lines into
`logs/results.csv` (row = setting, columns = per-phase seconds).

File naming: `tool_*` = experiment setup (not part of a task); `NN_*` = the
per-task SQL. **One prediction task = 4 SQL files** (01 label -> 02 features ->
03 task table -> 08 predict), plus `09` once at the end.

Manual order (reuse mode):
```
tool_cutoffs.sql                  (once, experiment setup)
01_label_adctr.sql      -v h=1/3/7 (per task)
tool_feat_cache_init.sql           (ONCE -> reuse ; per-task -> no-reuse)
02_features_adctr_pit.sql -v h     (per task: computes delta keys into cache)
03_task_table.sql       -v h       (per task: join label + cache)
08_predict_candidates.sql -v h -v sched=on|off -v cand=...   (per task)
09_action_list.sql      -v k=10    (once, after all horizons)
```

## Notes

- This rel-avito subsample spans only 2015-04-25..05-20 (25 days). Cutoffs are
  fixed to 2015-05-02..05-13 (>=7d history, >=7d future) so 1/3/7d are all valid;
  relbench's own test_timestamp (05-14) is NOT reused (05-14+7d overflows data).
- Source tables (`adsinfo`, `searchstream`, ...) live in db `avito`, loaded by
  `../../../.local/exp/load_avito.py`. See `../README.md` (dataset doc).
