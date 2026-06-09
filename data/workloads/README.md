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

## Run

```bash
bash run_avito.sh     # reuse mode, builds w_task_1/3/7 ; horizons via HORIZONS=...
bash bench.sh         # times reuse vs no-reuse feature compute + rows computed
```

File naming: `tool_*` = experiment setup (not part of a task); `NN_*` = the
per-task SQL. **One task = 3 SQL files** (01 label -> 02 features -> 03 task).

Manual order (reuse mode):
```
tool_cutoffs.sql                  (once, experiment setup)
01_label_adctr.sql      -v h=1/3/7 (per task)
tool_feat_cache_init.sql           (ONCE -> reuse ; per-task -> no-reuse)
02_features_adctr_pit.sql -v h     (per task: computes delta keys into cache)
03_task_table.sql       -v h       (per task: join label + cache)
```

## Notes

- This rel-avito subsample spans only 2015-04-25..05-20 (25 days). Cutoffs are
  fixed to 2015-05-02..05-13 (>=7d history, >=7d future) so 1/3/7d are all valid;
  relbench's own test_timestamp (05-14) is NOT reused (05-14+7d overflows data).
- Source tables (`adsinfo`, `searchstream`, ...) live in db `avito`, loaded by
  `../../.local/exp/load_avito.py`. See `../README.md`.
