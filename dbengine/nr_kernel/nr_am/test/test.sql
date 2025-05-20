DROP EXTENSION IF EXISTS ccam CASCADE;
CREATE EXTENSION ccam;
CREATE TABLE x(a INT, b text) USING ccam;

SELECT * from x;

DROP FUNCTION IF EXISTS run_nram_tests();
CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;
SELECT run_nram_tests();
