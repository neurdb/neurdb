CREATE EXTENSION nram;
CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;
SELECT run_nram_tests();
