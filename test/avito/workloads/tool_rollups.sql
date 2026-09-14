-- ============================================================================
-- tool_rollups.sql  --  Shared daily rollups (experiment setup, like cutoffs)
-- ============================================================================
-- Pre-aggregates the event streams to (adid, day) granularity; labels (01)
-- and PIT features (02) are then computed from these rollups instead of
-- re-scanning the raw streams. Together with the PIT feature cache this is
-- the shared data-prep artifact governed by the REUSE (CACHE) axis of the
-- ablation: reuse ON builds it once per run, reuse OFF rebuilds it before
-- every task (see run_tasks.sh).
--
-- Two exactness tricks:
--   * day bucket = date_trunc('day', evt_time - 1us), so bucket B holds events
--     in (B, B+1d]. Every window in this workload is half-open (lo, hi] with
--     midnight bounds, so any window maps EXACTLY onto a bucket range:
--       evt <= t            <=>  day <  t
--       evt in (t-7d, t]    <=>  day in [t-7d, t-1d]
--       evt in (t,   t+h]   <=>  day in [t,    t+h-1d]
--   * only ads with >= 1 click EVER can appear in a label (HAVING clicks > 0)
--     or in a feature key (keys come from labels), so the rollups are
--     restricted to w_clicked_ads up front.
--
-- All rollup sums/counts decompose the original aggregates exactly:
--   AVG(x) = SUM(sum_x) / SUM(n_x)   (NULLs excluded per-day, same as AVG).
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -f tool_rollups.sql
-- ============================================================================

SET work_mem = '512MB';
SET max_parallel_workers_per_gather = 8;

DROP TABLE IF EXISTS w_clicked_ads;
CREATE TABLE w_clicked_ads AS
SELECT adid
FROM searchstream
WHERE adid IS NOT NULL
GROUP BY adid
HAVING SUM(isclick) > 0;
ALTER TABLE w_clicked_ads ADD PRIMARY KEY (adid);

DROP TABLE IF EXISTS w_ss_daily;
CREATE TABLE w_ss_daily AS
SELECT s.adid,
       date_trunc('day', s.searchdate - INTERVAL '1 microsecond') AS day,
       COUNT(s.searchid) AS impr,
       SUM(s.isclick)    AS clicks,
       COUNT(s.isclick)  AS n_click,
       SUM(s.position)   AS sum_pos,
       COUNT(s.position) AS n_pos,
       SUM(s.histctr)    AS sum_histctr,
       COUNT(s.histctr)  AS n_histctr
FROM searchstream s
JOIN w_clicked_ads ca ON ca.adid = s.adid
GROUP BY s.adid, date_trunc('day', s.searchdate - INTERVAL '1 microsecond');
ALTER TABLE w_ss_daily ADD PRIMARY KEY (adid, day);

DROP TABLE IF EXISTS w_vs_daily;
CREATE TABLE w_vs_daily AS
SELECT v.adid,
       date_trunc('day', v.viewdate - INTERVAL '1 microsecond') AS day,
       COUNT(*) AS visits
FROM visitstream v
JOIN w_clicked_ads ca ON ca.adid = v.adid
GROUP BY v.adid, date_trunc('day', v.viewdate - INTERVAL '1 microsecond');
ALTER TABLE w_vs_daily ADD PRIMARY KEY (adid, day);

DROP TABLE IF EXISTS w_pr_daily;
CREATE TABLE w_pr_daily AS
SELECT p.adid,
       date_trunc('day', p.phonerequestdate - INTERVAL '1 microsecond') AS day,
       COUNT(*) AS reqs
FROM phonerequestsstream p
JOIN w_clicked_ads ca ON ca.adid = p.adid
GROUP BY p.adid, date_trunc('day', p.phonerequestdate - INTERVAL '1 microsecond');
ALTER TABLE w_pr_daily ADD PRIMARY KEY (adid, day);

\echo 'rollups:'
SELECT 'w_clicked_ads' AS rel, count(*) AS n_rows FROM w_clicked_ads
UNION ALL SELECT 'w_ss_daily', count(*) FROM w_ss_daily
UNION ALL SELECT 'w_vs_daily', count(*) FROM w_vs_daily
UNION ALL SELECT 'w_pr_daily', count(*) FROM w_pr_daily;
