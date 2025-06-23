BEGIN;
CREATE TABLE x(a INT, b text, c text, PRIMARY KEY (a, b)) USING nram;
-- Test insert and table scan.
SELECT * from x;
INSERT INTO x VALUES (1, 'k1', 'v1');
SELECT * from x;
COMMIT;

BEGIN;
SELECT * from x;
INSERT INTO x VALUES (2, 'k2', 'v2');
SELECT * from x;
DROP TABLE x;
COMMIT;
