DROP TABLE IF EXISTS x;
DROP TABLE IF EXISTS y;

CREATE TABLE x(a INT, b text, c text, PRIMARY KEY (a, b)) USING nram;

-- Test multi-insert and table scan.
INSERT INTO x VALUES (1, 'k1', 'v1'), (2, 'k2', 'v2'), (3, 'k3', 'v3');
SELECT * from x;
-- Test insert and table scan.
INSERT INTO x VALUES (1, 'k1', 'v1');
-- Drop all.
DROP TABLE x;

