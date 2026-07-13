-- ============================================================================
-- 01_label_adctr.sql  --  AdCTR LABEL table for one horizon  (FUTURE window)
-- ============================================================================
-- Horizon-dependent. Mirrors relbench AdCTRTask.make_table:
--   label(adid, t) = SUM(isclick) / COUNT(searchid)  over (t, t+h]
--   keep only ads that received >=1 click in the window (HAVING).
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

DROP TABLE IF EXISTS w_ad_label_:h;

CREATE TABLE w_ad_label_:h AS
SELECT s.adid,
       c.ts,
       SUM(s.isclick)::float8 / NULLIF(COUNT(s.searchid), 0) AS ctr,
       SUM(s.isclick)                                        AS clicks,
       COUNT(s.searchid)                                     AS impressions
FROM w_cutoffs c
JOIN searchstream s
  ON s.searchdate >  c.ts
 AND s.searchdate <= c.ts + (:h * INTERVAL '1 day')
GROUP BY s.adid, c.ts
HAVING SUM(s.isclick) > 0;

ALTER TABLE w_ad_label_:h ADD PRIMARY KEY (adid, ts);

\echo 'w_ad_label_':h':'
SELECT count(*) AS n_rows, count(DISTINCT adid) AS n_ads, count(DISTINCT ts) AS n_cutoffs
FROM w_ad_label_:h;
