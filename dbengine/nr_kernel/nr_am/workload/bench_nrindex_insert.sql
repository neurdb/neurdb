SET enable_seqscan = off;
\set iid random(1, 10000000)
\set newid random(20000001, 70000001)
INSERT INTO covid_nrindex (id, val) SELECT :newid, val FROM ik_idx WHERE id = :iid ON CONFLICT (id) DO NOTHING;
