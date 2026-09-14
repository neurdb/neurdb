-- ============================================================================
-- 04_predict_adctr.sql  --  Invoke the AI operator (TabPFN) on one task table
-- ============================================================================
-- Final step of the workload chain:
--   01 (label) -> 02 (PIT features) -> 03 (task table w_task_<h>) -> 04 (PREDICT)
--
-- This is the database-native AI operator: the executor scans w_task_<h>,
-- PUSHES the rows in batches to the AI engine over websocket; the engine
-- runs TabPFN (type-aware preprocess -> fit context -> stream-predict).
--
-- NOTE (in-sample): the current PREDICT grammar feeds the SAME FROM-subquery
-- to both the train-context phase and the inference phase, so this predicts
-- in-sample over all rows of w_task_<h>. For held-out eval, filter the
-- subquery to one split (e.g. WHERE split = 'test') or extend the operator to
-- take a separate prediction source.
--
-- Requires per-DB setup to be done once (see header of the runner script):
--   CREATE EXTENSION nr_pipeline;  + empty model/layer/router tables
--   SELECT insert_ai_engine('127.0.0.1', 8090);   -- AI engine endpoint
-- and the AI engine running with TabPFN available.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 04_predict_adctr.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

-- Batch sizing: the executor must send exactly ceil(n_rows / batch) batches,
-- and the engine is told nr_task_num_batches up front, so compute it here.
\set batch 4096

SELECT count(*) AS _n FROM w_task_:h \gset
SELECT ((:_n + :batch - 1) / :batch) AS _nbatch \gset

\echo 'w_task_':h':  rows=':_n'  batch=':batch'  num_batches=':_nbatch

SET nr_task_batch_size  TO :batch;
SET nr_task_num_batches TO :_nbatch;
SET nr_task_epoch       TO 1;

PREDICT VALUE OF label_ctr
FROM (SELECT * FROM w_task_:h) AS t
TRAIN tabpfn ON *;
