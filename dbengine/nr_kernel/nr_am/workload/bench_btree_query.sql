SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
\set kid random(1, 20000000)
SELECT * FROM covid_btree WHERE val = (SELECT val FROM qk_idx WHERE id = :kid) LIMIT 1;
