-- ============================================================================
-- 06_predict_nested_topk.sql  --  PREDICT nested inline in SQL (no staging)
-- ============================================================================
-- Demonstrates PREDICT as a *composable relational operator*: the statement
-- appears directly in an outer query's FROM clause and its output relation
-- (all input columns + a trailing float8 nr_pred column) is consumed by
-- ordinary SQL operators -- top-k via ORDER BY ... LIMIT, aggregation via
-- GROUP BY, and filters on the prediction itself -- in a single query, with
-- no intermediate table (contrast with 05, which stages via materialization).
--
--   SELECT ... FROM (PREDICT VALUE OF <col> FROM (...) TRAIN <m> ON *) p ...
--
-- Plan shape:  outer ops -> SubqueryScan -> NeurDB Predict -> input scan.
-- Predictions are computed once per query; quals on nr_pred are evaluated
-- above the operator (never pushed into it).
--
-- Requires the per-DB setup done once by the runner script (nr_pipeline
-- extension, AI-engine catalog entry) and the AI engine with TabPFN running.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 06_predict_nested_topk.sql
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
-- (1) PREDICT -> TOP-K, fully inline: highest predicted-CTR ads.
-- --------------------------------------------------------------------------
\echo '--- TOP-K ads by predicted CTR (inline PREDICT) ---'
SELECT adid, ts, categoryid, price, round(nr_pred::numeric, 6) AS pred_ctr
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
ORDER BY nr_pred DESC
LIMIT :k;

-- --------------------------------------------------------------------------
-- (2) PREDICT -> AGGREGATION, fully inline: mean predicted CTR per category.
-- --------------------------------------------------------------------------
\echo '--- AGGREGATION: mean predicted CTR per category (inline PREDICT) ---'
SELECT categoryid,
       count(*)                          AS n_ads,
       round(avg(nr_pred)::numeric, 6)   AS mean_pred_ctr,
       round(max(nr_pred)::numeric, 6)   AS max_pred_ctr
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
GROUP BY categoryid
ORDER BY mean_pred_ctr DESC
LIMIT :k;

-- --------------------------------------------------------------------------
-- (3) PREDICT -> FILTER + AGGREGATION: prediction-dependent selection.
--     The qual on nr_pred is evaluated above the AI operator.
-- --------------------------------------------------------------------------
\echo '--- FILTER on prediction: ads with predicted CTR > 0.2 ---'
SELECT count(*)                        AS high_ctr_ads,
       round(avg(price)::numeric, 2)   AS avg_price
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE nr_pred > 0.2;
