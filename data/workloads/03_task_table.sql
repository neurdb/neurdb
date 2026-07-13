-- ============================================================================
-- 03_task_table.sql  --  Final per-task table = LABEL (x) cached FEATURES
-- ============================================================================
-- Joins this task's label with the feature cache to produce the flat supervised
-- table fed to the AI operator (TabPFN):
--   w_task_<h> = w_ad_label_<h>  (x)  w_feat_cache   on (adid, ts)
-- One row = many aggregated feature columns + regression target label_ctr.
-- (The features it reads were either just computed by 04 or were cache hits.)
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 03_task_table.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

DROP TABLE IF EXISTS w_task_:h;

CREATE TABLE w_task_:h AS
SELECT f.*,
       l.ctr AS label_ctr
FROM w_ad_label_:h l
JOIN w_feat_cache f
  ON f.adid = l.adid AND f.ts = l.ts;

ALTER TABLE w_task_:h ADD PRIMARY KEY (adid, ts);

\echo 'w_task_':h' (split / rows / mean label):'
SELECT split, count(*) AS n_rows, round(avg(label_ctr)::numeric, 5) AS mean_ctr
FROM w_task_:h
GROUP BY split ORDER BY 1;
