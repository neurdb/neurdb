-- ============================================================================
-- 05_predict_topk_adctr.sql  --  PREDICT -> top-k / aggregation in one chain
-- ============================================================================
-- Demonstrates the AI operator as a *relation-producing* operator: PREDICT
-- materializes its output (every input row + a trailing nr_pred column) into a
-- target table, after which ordinary relational operators (top-k via
-- ORDER BY ... LIMIT, and aggregation via GROUP BY) consume the predictions
-- directly in SQL.
--
-- Mechanism: setting the session GUC `neurdb.predict_into` to a (pre-created)
-- table makes the PREDICT executor node append nr_pred to each scanned row and
-- write the augmented row into that table.  The target table must have exactly
-- (input columns + 1) columns: the FROM-subquery columns followed by a final
-- float8 prediction column.  Easiest to create it as `LIKE <src>, nr_pred`.
--
-- Chain:
--   01 (label) -> 02 (PIT features) -> 03 (task table w_task_<h>)
--               -> 04/05 (PREDICT -> materialize) -> top-k / aggregation
--
-- Requires the per-DB setup done once by the runner script:
--   CREATE EXTENSION nr_pipeline;  + empty model/layer/router tables
--   SELECT insert_ai_engine('127.0.0.1', 8090);
-- and the AI engine running with TabPFN available.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 05_predict_topk_adctr.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

\if :{?k}
\else
  \set k 10
\endif

-- Batch sizing: the executor sends exactly ceil(n_rows / batch) batches, and
-- the engine is told nr_task_num_batches up front, so compute it here.
\set batch 4096

SELECT count(*) AS _n FROM w_task_:h \gset
SELECT ((:_n + :batch - 1) / :batch) AS _nbatch \gset

\echo 'w_task_':h':  rows=':_n'  batch=':batch'  num_batches=':_nbatch'  top_k=':k

SET nr_task_batch_size  TO :batch;
SET nr_task_num_batches TO :_nbatch;
SET nr_task_epoch       TO 1;

-- --------------------------------------------------------------------------
-- (1) Materialization target: same columns as the FROM source, plus nr_pred.
--     No indexes are copied on purpose (the operator writes via a fast path
--     that does not maintain secondary indexes).
-- --------------------------------------------------------------------------
DROP TABLE IF EXISTS w_pred_:h;
CREATE TABLE w_pred_:h (LIKE w_task_:h, nr_pred double precision);

-- --------------------------------------------------------------------------
-- (2) AI operator: PREDICT -> materialize into w_pred_<h>.
--     Row output is suppressed here (rows are persisted to the table); we just
--     need them landed for the downstream relational operators.
--     NB: psql does not interpolate ':h' inside a quoted literal, so build the
--     table name into a variable and pass it with the :'var' quoting form.
-- --------------------------------------------------------------------------
\set pred_tbl w_pred_:h
SET neurdb.predict_into = :'pred_tbl';

\o /dev/null
PREDICT VALUE OF label_ctr
FROM (SELECT * FROM w_task_:h) AS t
TRAIN tabpfn ON *;
\o

RESET neurdb.predict_into;

\echo '--- materialized predictions: row count ---'
SELECT count(*) AS predicted_rows FROM w_pred_:h;

-- --------------------------------------------------------------------------
-- (3) PREDICT -> TOP-K : highest predicted-CTR ads.
-- --------------------------------------------------------------------------
\echo '--- TOP-K ads by predicted CTR ---'
SELECT adid, ts, categoryid, price, round(nr_pred::numeric, 6) AS pred_ctr
FROM w_pred_:h
ORDER BY nr_pred DESC
LIMIT :k;

-- --------------------------------------------------------------------------
-- (4) PREDICT -> AGGREGATION : mean predicted CTR per category (top categories).
-- --------------------------------------------------------------------------
\echo '--- AGGREGATION: mean predicted CTR per category (top categories) ---'
SELECT categoryid,
       count(*)                          AS n_ads,
       round(avg(nr_pred)::numeric, 6)   AS mean_pred_ctr,
       round(max(nr_pred)::numeric, 6)   AS max_pred_ctr
FROM w_pred_:h
GROUP BY categoryid
ORDER BY mean_pred_ctr DESC
LIMIT :k;

-- --------------------------------------------------------------------------
-- (5) PREDICT -> AGGREGATION : global summary of the prediction column.
-- --------------------------------------------------------------------------
\echo '--- AGGREGATION: global prediction summary ---'
SELECT count(*)                                          AS n,
       round(avg(nr_pred)::numeric, 6)                   AS mean_pred,
       round(min(nr_pred)::numeric, 6)                   AS min_pred,
       round(max(nr_pred)::numeric, 6)                   AS max_pred,
       round(stddev_samp(nr_pred)::numeric, 6)           AS std_pred
FROM w_pred_:h;
