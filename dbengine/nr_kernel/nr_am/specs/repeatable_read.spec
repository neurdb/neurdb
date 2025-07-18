setup
{
  SET client_min_messages TO WARNING;
  DROP TABLE IF EXISTS rc_test;
  DROP EXTENSION IF EXISTS nram CASCADE;
  DROP ACCESS METHOD IF EXISTS nram;
  DROP FUNCTION IF EXISTS nram_tableam_handler(internal);
  CREATE EXTENSION nram;
  CREATE TABLE accounts (id INT PRIMARY KEY, balance INT) USING nram;
  INSERT INTO accounts VALUES (1, 100);
}

teardown
{
  DROP TABLE accounts;
}

session s1
setup
{
  BEGIN ISOLATION LEVEL READ COMMITTED;
}
step s1_read1  { SELECT balance FROM accounts WHERE id = 1; }
step s1_read2  { SELECT balance FROM accounts WHERE id = 1; }
step s1_commit { COMMIT; }

session s2
setup
{
  BEGIN;
}
step s2_update { UPDATE accounts SET balance = 50 WHERE id = 1; }
step s2_commit { COMMIT; }

permutation s1_read1 s2_update s2_commit s1_read2 s1_commit
