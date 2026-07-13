-- ============================================================================
-- 02_features_adctr_pit.sql  --  Compute PIT features for ONE task, cache-aware
-- ============================================================================
-- Each prediction task (horizon :h) needs PIT features for its own label rows
-- w_ad_label_<h>. This computes them, but ONLY for the (adid, ts) keys that are
-- NOT already in the feature cache (w_feat_cache) -- cache hits are skipped.
-- Newly computed rows are appended to the cache.
--
-- Horizon-INDEPENDENT feature values: every aggregate uses only data <= t, so a
-- given (adid, t) has the same features no matter which task asked for it -> safe
-- to cache and reuse across horizons.
--
-- History aggregates are computed from the shared daily rollups
-- (tool_rollups.sql), whose buckets B hold events in (B, B+1d]:
--   evt <= t          <=>  day <  t
--   evt in (t-7d, t]  <=>  day >= t - 7d  (together with day < t)
-- Decompositions are exact, incl. NULL handling:
--   COUNT(x)  -> SUM(n_x),  SUM(x) -> SUM(sum_x),  AVG(x) -> SUM(sum_x)/SUM(n_x)
--
-- Mode is decided by the caller (see tool_feat_cache_init.sql):
--   * cache ON : the "INSERT 0 N" line shows only the DELTA each task computed.
--   * cache OFF: cache was reset first, so N = all of this task's rows.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -v h=1 -f 02_features_adctr_pit.sql
-- ============================================================================

\if :{?h}
\else
  \set h 1
\endif

SET work_mem = '512MB';
SET max_parallel_workers_per_gather = 8;

INSERT INTO w_feat_cache (
    adid, ts, split,
    price, iscontext, categoryid, locationid, title_len,
    cat_level, cat_parent, loc_region, loc_city,
    ss_impr_all, ss_click_all, ss_ctr_all, ss_avgpos_all, ss_avghistctr_all,
    ss_impr_7d, ss_click_7d,
    vs_visit_all, vs_visit_7d, pr_all
)
WITH k AS (   -- this task's keys MINUS whatever is already cached (the delta)
    SELECT adid, ts FROM w_ad_label_:h
    EXCEPT
    SELECT adid, ts FROM w_feat_cache
),
ss AS (
    SELECT k.adid, k.ts,
           COALESCE(SUM(d.impr), 0)::bigint                                   AS ss_impr_all,
           COALESCE(SUM(d.clicks), 0)::float8                                 AS ss_click_all,
           (SUM(d.clicks) / NULLIF(SUM(d.n_click), 0)::float8)                AS ss_ctr_all,
           (SUM(d.sum_pos)::float8 / NULLIF(SUM(d.n_pos), 0)::float8)         AS ss_avgpos_all,
           (SUM(d.sum_histctr) / NULLIF(SUM(d.n_histctr), 0)::float8)         AS ss_avghistctr_all,
           COALESCE(SUM(d.impr)   FILTER (WHERE d.day >= k.ts - INTERVAL '7 day'), 0)::bigint AS ss_impr_7d,
           COALESCE(SUM(d.clicks) FILTER (WHERE d.day >= k.ts - INTERVAL '7 day'), 0)::float8 AS ss_click_7d
    FROM k
    LEFT JOIN w_ss_daily d
           ON d.adid = k.adid AND d.day < k.ts
    GROUP BY k.adid, k.ts
),
vs AS (
    SELECT k.adid, k.ts,
           COALESCE(SUM(d.visits), 0)::bigint                                 AS vs_visit_all,
           COALESCE(SUM(d.visits) FILTER (WHERE d.day >= k.ts - INTERVAL '7 day'), 0)::bigint AS vs_visit_7d
    FROM k
    LEFT JOIN w_vs_daily d
           ON d.adid = k.adid AND d.day < k.ts
    GROUP BY k.adid, k.ts
),
pr AS (
    SELECT k.adid, k.ts,
           COALESCE(SUM(d.reqs), 0)::bigint AS pr_all
    FROM k
    LEFT JOIN w_pr_daily d
           ON d.adid = k.adid AND d.day < k.ts
    GROUP BY k.adid, k.ts
)
SELECT k.adid, k.ts, c.split,
       a.price, a.iscontext, a.categoryid, a.locationid,
       length(a.title)        AS title_len,
       cat.level              AS cat_level,
       cat.parentcategoryid   AS cat_parent,
       loc.regionid           AS loc_region,
       loc.cityid             AS loc_city,
       ss.ss_impr_all, ss.ss_click_all, ss.ss_ctr_all,
       ss.ss_avgpos_all, ss.ss_avghistctr_all,
       ss.ss_impr_7d, ss.ss_click_7d,
       vs.vs_visit_all, vs.vs_visit_7d, pr.pr_all
FROM k
JOIN      w_cutoffs c   ON c.ts = k.ts
JOIN      adsinfo   a   ON a.adid = k.adid
LEFT JOIN category  cat ON cat.categoryid = a.categoryid
LEFT JOIN location  loc ON loc.locationid = a.locationid
LEFT JOIN ss ON ss.adid = k.adid AND ss.ts = k.ts
LEFT JOIN vs ON vs.adid = k.adid AND vs.ts = k.ts
LEFT JOIN pr ON pr.adid = k.adid AND pr.ts = k.ts;

\echo '   (INSERT count above = rows actually computed for h=':h'; cache size now:)'
SELECT count(*) AS cache_rows FROM w_feat_cache;
