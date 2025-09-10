setup
{
  SET client_min_messages TO WARNING;
  DROP TABLE IF EXISTS accounts;
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
setup { BEGIN ISOLATION LEVEL SERIALIZABLE; }
step s1_count1 { SELECT * FROM accounts WHERE balance >= 100; }
step s1_count2 { SELECT * FROM accounts WHERE balance >= 100; }
step s1_commit { COMMIT; }

session s2
setup { BEGIN ISOLATION LEVEL SERIALIZABLE; }
step s2_insert { INSERT INTO accounts VALUES (2, 200); }
step s2_commit { COMMIT; }

# Expect s1_count1=1 then s1_count2=2 under RC (phantom). Under SERIALIZABLE, one tx may be forced to retry.
permutation s1_count1 s2_insert s2_commit s1_count2 s1_commit
