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
           COUNT(s.searchid)                                                      AS ss_impr_all,
           COALESCE(SUM(s.isclick), 0)                                            AS ss_click_all,
           AVG(s.isclick)                                                         AS ss_ctr_all,
           AVG(s.position)                                                        AS ss_avgpos_all,
           AVG(s.histctr)                                                         AS ss_avghistctr_all,
           COUNT(s.searchid) FILTER (WHERE s.searchdate > k.ts - INTERVAL '7 day') AS ss_impr_7d,
           COALESCE(SUM(s.isclick) FILTER (WHERE s.searchdate > k.ts - INTERVAL '7 day'), 0) AS ss_click_7d
    FROM k
    LEFT JOIN searchstream s
           ON s.adid = k.adid AND s.searchdate <= k.ts
    GROUP BY k.adid, k.ts
),
vs AS (
    SELECT k.adid, k.ts,
           COUNT(v.adid)                                                         AS vs_visit_all,
           COUNT(v.adid) FILTER (WHERE v.viewdate > k.ts - INTERVAL '7 day')      AS vs_visit_7d
    FROM k
    LEFT JOIN visitstream v
           ON v.adid = k.adid AND v.viewdate <= k.ts
    GROUP BY k.adid, k.ts
),
pr AS (
    SELECT k.adid, k.ts,
           COUNT(p.adid) AS pr_all
    FROM k
    LEFT JOIN phonerequestsstream p
           ON p.adid = k.adid AND p.phonerequestdate <= k.ts
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
