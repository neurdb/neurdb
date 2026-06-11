-- ============================================================================
-- 01_label_adctr.sql  --  AdCTR LABEL table for one horizon  (FUTURE window)
-- ============================================================================
-- Horizon-dependent. Mirrors relbench AdCTRTask.make_table:
--   label(adid, t) = SUM(isclick) / COUNT(searchid)  over (t, t+h]
--   keep only ads that received >=1 click in the window (HAVING).
--
-- Computed from the shared daily rollup (tool_rollups.sql) instead of the raw
-- stream: the window (t, t+h] maps exactly onto rollup buckets [t, t+h-1d]
-- (see the bucket definition there), so we expand the 12 cutoffs x h days into
-- (ts, day) pairs and equi-join them to the rollup -- a hash join over ~100K
-- pre-aggregated rows instead of a range join over the 9.25M-row raw stream.
--
-- Parameterized by :h (horizon in days). Produces table  w_ad_label_<h>.
-- Run once per horizon, e.g.:
--   psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 01_label_adctr.sql
--   psql ... -d avito -v ON_ERROR_STOP=1 -v h=3 -f 01_label_adctr.sql
--   psql ... -d avito -v ON_ERROR_STOP=1 -v h=7 -f 01_label_adctr.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

SET work_mem = '512MB';
SET max_parallel_workers_per_gather = 8;

DROP TABLE IF EXISTS w_ad_label_:h;

CREATE TABLE w_ad_label_:h AS
SELECT d.adid,
       p.ts,
       SUM(d.clicks)::float8 / NULLIF(SUM(d.impr), 0)::float8 AS ctr,
       SUM(d.clicks)                                          AS clicks,
       SUM(d.impr)::bigint                                    AS impressions
FROM (SELECT c.ts, c.ts + (g.i || ' day')::interval AS day
      FROM w_cutoffs c, generate_series(0, :h - 1) AS g(i)) p
JOIN w_ss_daily d ON d.day = p.day
GROUP BY d.adid, p.ts
HAVING SUM(d.clicks) > 0;

ALTER TABLE w_ad_label_:h ADD PRIMARY KEY (adid, ts);

\echo 'w_ad_label_':h':'
SELECT count(*) AS n_rows, count(DISTINCT adid) AS n_ads, count(DISTINCT ts) AS n_cutoffs
FROM w_ad_label_:h;
