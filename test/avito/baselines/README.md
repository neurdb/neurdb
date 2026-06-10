# LOTUS / Palimpzest baselines (export-execute-import)

External-framework baselines for the avito AdCTR horizon-sweep NLQ
(`test/avito/workloads/`), addressing the reviewer request to compare against
LOTUS and Palimpzest. Both run ON THE HOST (not in docker), each in its own
conda env, against data exported once from NeurDB -- the classic
export-execute-import pipeline the paper argues against.

## What is identical to NeurEngine

* **Workload**: same NLQ -> 3 PREDICT tasks (horizons 1/3/7 days), same
  candidate set (`categoryid IN (60, 26, 27)`), same action-list aggregation.
* **Data semantics**: labels, PIT features and task tables replicate
  `01_label_adctr.sql` / `02_features_adctr_pit.sql` / `03_task_table.sql`
  row-for-row (verified: `w_task_1` = 8368 rows in both).
* **Model + preprocessing**: the scripts import the engine's own
  `TabularPreprocessor` / `StatefulTabPFN` modules
  (`aiengine/runtime/neurdbrt/model/tabpfn/`), with the same pg-type stype
  hints, context limit (10k) and batch size (4096), running in-context fit on
  the candidate rows -- equivalent to NeurEngine's pushed-down PREDICT.

## What differs (the point of the experiment)

* Data leaves the DB (timed `export_*` steps).
* Each horizon task recomputes its features from the raw exported tables --
  neither framework has cross-task materialization/reuse.
* Neither framework has a tabular-prediction operator:
  * **LOTUS** `sem_*` operators are LLM/embedding-only; the pipeline is its
    pandas substrate with TabPFN as opaque user code outside any optimizer.
  * **Palimpzest** runs the per-task prediction as a `Dataset` pipeline
    (`filter` -> `map(udf)`, MinCost policy), but its UDF contract is one
    record at a time; the UDF fits the TabPFN context once, batch-predicts and
    memoizes -- the most favorable integration its interface allows.

## Files

| file | role |
|---|---|
| `setup_envs.sh` | create conda envs `bl_lotus` (py3.11) / `bl_pz` (py3.12), torch 2.4.1 cu118 + tabpfn 2.2.1 + framework |
| `export_data.py` | COPY the 6 needed tables out of NeurDB into `data/*.parquet` |
| `avito_pipeline.py` | shared core: cutoffs/label/PIT-features/task/action-list + TabPFN bridge |
| `run_lotus.py` | LOTUS baseline runner |
| `run_palimpzest.py` | Palimpzest baseline runner |
| `run_baselines.sh` | driver: export once + run both + aggregate `logs/baseline_results.csv` |

## Run

```bash
bash setup_envs.sh all        # once
bash run_baselines.sh all     # logs/<system>.log + logs/baseline_results.csv
```

The DB must accept host connections (`listen_addresses='*'` +
`host all all 0.0.0.0/0 trust` in `build/psql/data/pg_hba.conf`).

Compare against the NeurEngine numbers from
`test/avito/workloads/run_matrix.sh` (`logs/results.csv`).
