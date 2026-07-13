-- ============================================================================
-- tool_cutoffs.sql  --  Prediction-time grid (experiment setup, shared by all tasks)
-- ============================================================================
-- A "cutoff" t is a prediction time. For each ad we build:
--   * FEATURES  from history  (.. , t]      -- horizon-independent (reusable)
--   * LABEL     from future   (t , t+h]     -- horizon-dependent
--
-- The grid is FIXED and horizon-independent on purpose: every horizon variant
-- (1d/3d/7d) uses the same cutoffs, so the PIT feature table can be built once
-- and reused across horizons.
--
-- Data range of this rel-avito 100k subsample: 2015-04-25 .. 2015-05-20 (25d).
-- We reserve >=7d of history at the start and >=7d of future at the end so the
-- largest horizon (7d) always has a valid label window. (relbench's own
-- test_timestamp=2015-05-14 + 7d would overflow the data, so we do NOT reuse it.)
--
-- Run (inside container):
--   psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1 -f tool_cutoffs.sql
-- ============================================================================

DROP TABLE IF EXISTS w_cutoffs;

CREATE TABLE w_cutoffs AS
WITH bounds AS (
    SELECT date_trunc('day', min(searchdate)) AS mn,
           date_trunc('day', max(searchdate)) AS mx
    FROM searchstream
),
grid AS (
    SELECT gs AS ts
    FROM bounds,
         generate_series(mn + INTERVAL '7 day',   -- >=7d history for features
                         mx - INTERVAL '7 day',    -- >=7d future for max horizon
                         INTERVAL '1 day') AS gs
)
SELECT ts,
       CASE WHEN ts <  DATE '2015-05-11' THEN 'train'
            WHEN ts <  DATE '2015-05-13' THEN 'val'
            ELSE 'test'
       END AS split
FROM grid
ORDER BY ts;

ALTER TABLE w_cutoffs ADD PRIMARY KEY (ts);

\echo 'w_cutoffs:'
SELECT split, count(*) AS n_cutoffs, min(ts), max(ts) FROM w_cutoffs GROUP BY split ORDER BY 1;
