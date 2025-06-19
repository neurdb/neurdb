setup
{
  SET client_min_messages TO WARNING;
  DROP TABLE IF EXISTS rc_test;
  DROP EXTENSION IF EXISTS nram CASCADE;
  DROP ACCESS METHOD IF EXISTS nram;
  DROP FUNCTION IF EXISTS nram_tableam_handler(internal);
  CREATE EXTENSION nram;
  CREATE TABLE rc_test (id INT PRIMARY KEY, val TEXT) USING nram;
  INSERT INTO rc_test VALUES (1, 'original');
}

teardown
{
  DROP TABLE rc_test;
}

session s1
setup
{
  BEGIN ISOLATION LEVEL READ COMMITTED;
}
step s1_update { UPDATE rc_test SET val = 'updated' WHERE id = 1; }
step s1_commit { COMMIT; }

session s2
setup
{
  BEGIN ISOLATION LEVEL READ COMMITTED;
}
step s2_read1 { SELECT * FROM rc_test WHERE id = 1; }
step s2_commit { COMMIT; }

permutation s1_update s2_read1 s1_commit s2_commit
