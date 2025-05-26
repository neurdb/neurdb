CREATE EXTENSION nram;
CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;
SELECT run_nram_tests();

CREATE TABLE x(a INT, b text, c text, PRIMARY KEY (a, b)) USING nram;
-- Insert two test rows
INSERT INTO x VALUES (1, 'k1', 'v1');
SELECT * from x;

-- -- SELECT * from x;
-- -- INSERT INTO x VALUES (2, 'k2', 'v2');

