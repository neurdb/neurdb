-- predict_topk.sql -- candidate-scoped PREDICT + top-k (no table materialization)
--
-- Auto-computes nr_task_num_batches from rows actually fed to the operator
-- (respects nr_predict_pushdown / sched mode).
--
-- Run (inside container):
--   psql ... -d avito -v ON_ERROR_STOP=1 \
--     -v h=1 -v sched=on -v cand="categoryid IN (60, 26, 27)" -v k=10 \
--     -f predict_topk.sql
--
-- Or from host:
--   bash predict_topk.sh

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
  \set cand categoryid IN (60, 26, 27)
\endif

\if :{?k}
\else
  \set k 10
\endif

\if :{?batch}
\else
  \set batch 4096
\endif

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

\echo '=== predict_topk  h=':h'  sched=':sched'  cand=':cand
\echo '    rows total=':_n_all'  candidate=':_n_cand'  fed=':_n_fed'  batch=':batch'  nb=':_nb

SET nr_task_batch_size  TO :batch;
SET nr_task_num_batches TO :_nb;
SET nr_task_epoch       TO 1;

\timing on
SELECT adid, ts, categoryid, round(nr_pred::numeric, 6) AS pred_ctr
FROM (PREDICT VALUE OF label_ctr
      FROM (SELECT * FROM w_task_:h) AS s
      TRAIN tabpfn ON *) AS p
WHERE :cand
ORDER BY nr_pred DESC
LIMIT :k;
\timing off

RESET nr_predict_pushdown;
