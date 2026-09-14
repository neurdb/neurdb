-- ============================================================================
-- 08_predict_candidates.sql -- per-horizon candidate-scoped PREDICT -> w_pred_<h>
-- ============================================================================
-- The per-prediction-task "AI operator + ranking + persist" step of the
-- horizon-sweep workload (exp.md section 5, step 4-5), in ONE statement:
--
--   CREATE TABLE w_pred_<h> AS
--   SELECT ... FROM (PREDICT VALUE OF label_ctr FROM w_task_<h> ...) p
--   WHERE <candidate predicate>;
--
-- The candidate predicate (NLQ: "I have a set of candidate ads to promote")
-- is the optimizer's scheduling decision point, toggled by :sched:
--
--   sched=off  BASELINE: operator pinned at the root of its subquery; the
--              candidate qual is evaluated ABOVE it -> every row of
--              w_task_<h> is trained on and inferred.
--   sched=on   DYNAMIC: the planner pushes the input-column qual BELOW the
--              AI operator -> only candidate rows are fetched, trained on
--              and inferred.  (Quals on nr_pred can never sink below.)
--
-- Both modes produce the same w_pred_<h> relation (candidate rows + nr_pred);
-- predictions differ only via the in-context training set.  The EXPLAIN
-- printed before the run shows where the planner scheduled the operator and
-- the cost model's (cost_neurdbpredict) pricing of the plan.
--
-- NB: nr_task_num_batches must equal ceil(rows_actually_fed / batch): the
-- full count when sched=off, the candidate count when sched=on.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 \
--            -v h=1 -v sched=on -v cand="categoryid IN (60, 26, 27)" \
--            -f 08_predict_candidates.sql
-- (cand must not contain single quotes; default candidate set below.)
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

\if :{?sched}
\else
  \set sched on
\endif

\if :{?cand}
\else
  \set cand 'categoryid IN (60, 26, 27)'
\endif

\if :{?k}
\else
  \set k 5
\endif

\set batch 4096

SELECT count(*) AS _n_all  FROM w_task_:h \gset
SELECT count(*) AS _n_cand FROM w_task_:h WHERE :cand \gset

\if :sched
  SET nr_predict_pushdown = on;
  \set _n_fed :_n_cand
\else
  SET nr_predict_pushdown = off;
  \set _n_fed :_n_all
\endif

SELECT ((:_n_fed + :batch - 1) / :batch) AS _nb \gset

\echo '=== 08 h=':h'  sched=':sched'  candidates: ':cand
\echo '    rows total=':_n_all'  candidate=':_n_cand'  fed_to_operator=':_n_fed'  batches=':_nb

SET nr_task_batch_size  TO :batch;
SET nr_task_num_batches TO :_nb;
SET nr_task_epoch       TO 1;

-- Where did the planner schedule the AI operator? (filter below vs above it)
EXPLAIN
SELECT adid, ts, categoryid, price, label_ctr, nr_pred
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE :cand;

DROP TABLE IF EXISTS w_pred_:h;

\timing on
CREATE TABLE w_pred_:h AS
SELECT adid, ts, categoryid, price, label_ctr, nr_pred
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE :cand;
\timing off

ALTER TABLE w_pred_:h ADD PRIMARY KEY (adid, ts);

\echo '--- w_pred_':h' summary ---'
SELECT count(*)                                              AS n_rows,
       round(avg(nr_pred)::numeric, 6)                       AS mean_pred,
       round(sqrt(avg((nr_pred - label_ctr)^2))::numeric, 6) AS rmse
FROM w_pred_:h;

\echo '--- top-':k' candidate ads by predicted CTR (h=':h') ---'
SELECT adid, ts, categoryid, price, round(nr_pred::numeric, 6) AS pred_ctr
FROM w_pred_:h
ORDER BY nr_pred DESC
LIMIT :k;

RESET nr_predict_pushdown;
