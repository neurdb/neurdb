CREATE TABLE x(a INT, b text, c text, PRIMARY KEY (a, b)) USING nram;
CREATE TABLE y(a INT, b text, PRIMARY KEY (a)) USING nram;

-- Test insert and table scan.
INSERT INTO x VALUES (1, 'k1', 'v1');
SELECT * from x;
INSERT INTO x VALUES (2, 'k2', 'v2');
SELECT * from x;
-- Test different tables.
INSERT INTO y VALUES (1, 'z1');
INSERT INTO y VALUES (2, 'z2');
SELECT * from x;
SELECT * from y;
-- Drop all.
DROP TABLE x;
DROP TABLE y;
