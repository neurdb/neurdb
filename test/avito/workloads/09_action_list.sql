-- ============================================================================
-- 09_action_list.sql -- aggregate the horizon sweep into ONE action list
-- ============================================================================
-- The final task of the horizon-sweep workload (exp.md section 5, last step):
-- join the per-horizon prediction tables w_pred_1 / w_pred_3 / w_pred_7
-- (built by 08) on (adid, ts) and translate the prediction PROFILE of each
-- candidate ad into an operational action:
--
--   promote_now    high short-term CTR (1d in top-:k of its cutoff)
--   promote_later  weak now but strong later (7d in top-:k, 1d not)
--   keep           neither extreme
--   reduce_exposure weak across ALL horizons (bottom quartile everywhere)
--
-- Rankings are computed per cutoff ts, so "top-k" means "top-k of that day's
-- candidate slate" -- the slate an ad ops team would actually act on.
--
-- Only ads present in all three task tables are scored (inner join): an ad
-- must be labelable at every horizon to get a full profile.
--
-- Horizons are fixed to 1/3/7 to match the experiment task list.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v k=10 -f 09_action_list.sql
-- ============================================================================

\if :{?k}
\else
  \set k 10
\endif

DROP TABLE IF EXISTS w_action_list;

CREATE TABLE w_action_list AS
WITH joined AS (
    SELECT p1.adid, p1.ts, p1.categoryid, p1.price,
           p1.nr_pred AS ctr_1d,
           p3.nr_pred AS ctr_3d,
           p7.nr_pred AS ctr_7d
    FROM w_pred_1 p1
    JOIN w_pred_3 p3 USING (adid, ts)
    JOIN w_pred_7 p7 USING (adid, ts)
),
ranked AS (
    SELECT *,
           rank()  OVER (PARTITION BY ts ORDER BY ctr_1d DESC) AS rk_1d,
           rank()  OVER (PARTITION BY ts ORDER BY ctr_3d DESC) AS rk_3d,
           rank()  OVER (PARTITION BY ts ORDER BY ctr_7d DESC) AS rk_7d,
           ntile(4) OVER (PARTITION BY ts ORDER BY ctr_1d) AS q_1d,
           ntile(4) OVER (PARTITION BY ts ORDER BY ctr_3d) AS q_3d,
           ntile(4) OVER (PARTITION BY ts ORDER BY ctr_7d) AS q_7d
    FROM joined
)
SELECT adid, ts, categoryid, price,
       ctr_1d, ctr_3d, ctr_7d,
       rk_1d, rk_3d, rk_7d,
       CASE
           WHEN rk_1d <= :k                                THEN 'promote_now'
           WHEN rk_7d <= :k                                THEN 'promote_later'
           WHEN q_1d = 1 AND q_3d = 1 AND q_7d = 1         THEN 'reduce_exposure'
           ELSE 'keep'
       END AS action
FROM ranked;

ALTER TABLE w_action_list ADD PRIMARY KEY (adid, ts);

\echo '=== 09 action list (k=':k') ==='
SELECT action, count(*) AS n_ads,
       round(avg(ctr_1d)::numeric, 6) AS avg_ctr_1d,
       round(avg(ctr_7d)::numeric, 6) AS avg_ctr_7d
FROM w_action_list
GROUP BY action
ORDER BY CASE action
             WHEN 'promote_now' THEN 1
             WHEN 'promote_later' THEN 2
             WHEN 'keep' THEN 3
             ELSE 4
         END;

\echo '--- sample: promote_now ---'
SELECT adid, ts, categoryid,
       round(ctr_1d::numeric, 6) AS ctr_1d,
       round(ctr_3d::numeric, 6) AS ctr_3d,
       round(ctr_7d::numeric, 6) AS ctr_7d
FROM w_action_list WHERE action = 'promote_now'
ORDER BY ctr_1d DESC LIMIT 5;

\echo '--- sample: promote_later (weak 1d, strong 7d) ---'
SELECT adid, ts, categoryid,
       round(ctr_1d::numeric, 6) AS ctr_1d,
       round(ctr_3d::numeric, 6) AS ctr_3d,
       round(ctr_7d::numeric, 6) AS ctr_7d
FROM w_action_list WHERE action = 'promote_later'
ORDER BY ctr_7d DESC LIMIT 5;

\echo '--- sample: reduce_exposure ---'
SELECT adid, ts, categoryid,
       round(ctr_1d::numeric, 6) AS ctr_1d,
       round(ctr_3d::numeric, 6) AS ctr_3d,
       round(ctr_7d::numeric, 6) AS ctr_7d
FROM w_action_list WHERE action = 'reduce_exposure'
ORDER BY ctr_1d ASC LIMIT 5;
