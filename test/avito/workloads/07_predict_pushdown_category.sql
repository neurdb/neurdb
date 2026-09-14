-- ============================================================================
-- 07_predict_pushdown_category.sql -- dynamic AI-operator scheduling vs root
-- ============================================================================
-- Experiment: the same SQL -- "predicted CTR stats / top-k for ONE category" --
-- planned two ways, toggled by the nr_predict_pushdown GUC:
--
--   BASELINE (off):  the PREDICT operator is pinned at the root of its
--                    subquery; the category filter is evaluated ABOVE it, so
--                    every row of w_task_<h> is trained on and inferred.
--
--       SubqueryScan p  (Filter: categoryid = :cat)
--         -> NeurDB Predict          rows = |w_task|
--              -> Seq Scan w_task
--
--   DYNAMIC (on):    the planner pushes the input-column qual BELOW the AI
--                    operator (it never pushes quals on nr_pred); only the
--                    category's rows are fetched, trained on and inferred.
--                    The cost model (cost_neurdbpredict) prices the operator
--                    per row + per batch, which is what makes the pushed plan
--                    visibly cheaper in EXPLAIN.
--
--       SubqueryScan p
--         -> NeurDB Predict          rows = |category|
--              -> Seq Scan w_task    (Filter: categoryid = :cat)
--
-- Measured on w_task_1 (8368 rows), categoryid = 60 (398 rows, 4.8%):
--   baseline 13.1 s,  dynamic 2.9 s  (4.5x), RMSE 0.0933 vs 0.0858 -- the
--   dynamic plan also trains the in-context model on the category itself.
--
-- NB: nr_task_num_batches must equal ceil(rows_actually_fed / batch); with
-- pushdown on, that is the *filtered* row count.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -v cat=60 \
--            -f 07_predict_pushdown_category.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

\if :{?cat}
\else
  \set cat 60
\endif

\if :{?k}
\else
  \set k 10
\endif

\set batch 4096

SELECT count(*) AS _n_all FROM w_task_:h \gset
SELECT count(*) AS _n_cat FROM w_task_:h WHERE categoryid = :cat \gset
SELECT ((:_n_all + :batch - 1) / :batch) AS _nb_all \gset
SELECT ((:_n_cat + :batch - 1) / :batch) AS _nb_cat \gset

\echo 'w_task_':h':  rows=':_n_all'  category=':cat'  cat_rows=':_n_cat'  batch=':batch

SET nr_task_batch_size TO :batch;
SET nr_task_epoch      TO 1;

\timing on

-- --------------------------------------------------------------------------
-- (1) BASELINE: operator pinned at subquery root (no pushdown).
-- --------------------------------------------------------------------------
\echo '--- BASELINE (nr_predict_pushdown=off): infer all ':_n_all' rows ---'
SET nr_predict_pushdown = off;
SET nr_task_num_batches TO :_nb_all;

EXPLAIN
SELECT count(*), avg(nr_pred)
FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE categoryid = :cat;

SELECT count(*)                                               AS n_rows,
       round(avg(nr_pred)::numeric, 6)                        AS mean_pred,
       round(sqrt(avg((nr_pred - label_ctr)^2))::numeric, 6)  AS rmse
FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE categoryid = :cat;

-- --------------------------------------------------------------------------
-- (2) DYNAMIC: cost-based scheduling pushes the filter below the operator.
-- --------------------------------------------------------------------------
\echo '--- DYNAMIC (nr_predict_pushdown=on): infer only ':_n_cat' rows ---'
SET nr_predict_pushdown = on;
SET nr_task_num_batches TO :_nb_cat;

EXPLAIN
SELECT count(*), avg(nr_pred)
FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE categoryid = :cat;

SELECT count(*)                                               AS n_rows,
       round(avg(nr_pred)::numeric, 6)                        AS mean_pred,
       round(sqrt(avg((nr_pred - label_ctr)^2))::numeric, 6)  AS rmse
FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE categoryid = :cat;

-- --------------------------------------------------------------------------
-- (3) Top-k inside the category, dynamic plan: filter below, ORDER BY above.
--     (A qual/sort on nr_pred can never sink below the operator.)
-- --------------------------------------------------------------------------
\echo '--- TOP-':k' ads by predicted CTR within category ':cat' (dynamic) ---'
SELECT adid, ts, price, round(nr_pred::numeric, 6) AS pred_ctr
FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE categoryid = :cat
ORDER BY nr_pred DESC
LIMIT :k;

\timing off
RESET nr_predict_pushdown;
