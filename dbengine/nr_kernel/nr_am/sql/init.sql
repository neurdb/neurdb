CREATE EXTENSION nram;

CREATE OR REPLACE FUNCTION nram_load_policy(path text)
RETURNS void
AS 'nram', 'nram_load_policy'
LANGUAGE C
STRICT
VOLATILE
PARALLEL RESTRICTED;

CREATE OR REPLACE FUNCTION run_nram_tests()
RETURNS void
AS 'nram', 'run_nram_tests'
LANGUAGE C STRICT;
