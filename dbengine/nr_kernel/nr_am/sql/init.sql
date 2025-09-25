CREATE EXTENSION nram;
CREATE OR REPLACE FUNCTION nram_load_policy()
RETURNS void
AS 'nram', 'nram_load_policy'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;
