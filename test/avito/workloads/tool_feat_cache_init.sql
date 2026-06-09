-- ============================================================================
-- tool_feat_cache_init.sql  --  (re)create an EMPTY point-in-time feature cache
-- ============================================================================
-- The feature cache stores already-computed PIT features keyed by (adid, ts).
-- It is the artifact that makes feature work reusable ACROSS prediction tasks:
--   * a task only computes the (adid, ts) rows NOT already in the cache (04),
--   * cache hits are skipped.
--
-- This script just RESETS the cache to empty. How you call it decides the mode:
--   * REUSE  (cache on):  run this ONCE, then run all tasks -> cache accumulates,
--                         each task only computes its delta.
--   * NO-REUSE (cache off): run this BEFORE EVERY task -> cache always empty,
--                         each task recomputes all of its own rows from scratch.
--
-- Run:  psql ... -d avito -v ON_ERROR_STOP=1 -f tool_feat_cache_init.sql
-- ============================================================================

DROP TABLE IF EXISTS w_feat_cache;

CREATE TABLE w_feat_cache (
    adid              BIGINT,
    ts                TIMESTAMP,
    split             TEXT,
    -- static ad attributes
    price             DOUBLE PRECISION,
    iscontext         DOUBLE PRECISION,
    categoryid        BIGINT,
    locationid        BIGINT,
    title_len         INTEGER,
    cat_level         BIGINT,
    cat_parent        BIGINT,
    loc_region        DOUBLE PRECISION,
    loc_city          DOUBLE PRECISION,
    -- searchstream history (<= t)
    ss_impr_all       BIGINT,
    ss_click_all      DOUBLE PRECISION,
    ss_ctr_all        DOUBLE PRECISION,
    ss_avgpos_all     DOUBLE PRECISION,
    ss_avghistctr_all DOUBLE PRECISION,
    ss_impr_7d        BIGINT,
    ss_click_7d       DOUBLE PRECISION,
    -- visit / phone history (<= t)
    vs_visit_all      BIGINT,
    vs_visit_7d       BIGINT,
    pr_all            BIGINT,
    PRIMARY KEY (adid, ts)
);
