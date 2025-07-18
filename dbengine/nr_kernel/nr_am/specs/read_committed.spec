setup
{
  SET client_min_messages TO WARNING;
  DROP TABLE IF EXISTS rc_test;
  DROP EXTENSION IF EXISTS nram CASCADE;
  DROP ACCESS METHOD IF EXISTS nram;
  DROP FUNCTION IF EXISTS nram_tableam_handler(internal);
  CREATE EXTENSION nram;
  CREATE TABLE rc_test (id INT PRIMARY KEY, val TEXT) USING nram;
}

teardown
{
  DROP TABLE rc_test;
}

session s1
step s1_begin { BEGIN ISOLATION LEVEL READ COMMITTED; }
step s1_insert { INSERT INTO rc_test VALUES (1, 'original');}
step s1_read1 { SELECT * FROM rc_test; }
step s1_commit { COMMIT; }

session s2
step s2_begin { BEGIN ISOLATION LEVEL READ COMMITTED; }
step s2_read1 { SELECT * FROM rc_test; }
step s2_commit { COMMIT; }

permutation s1_begin s1_insert s1_read1 s1_commit s2_begin s2_read1 s2_commit s1_begin s1_read1 s1_commit
