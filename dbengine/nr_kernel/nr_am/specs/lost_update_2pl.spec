setup
{
  SET client_min_messages TO WARNING;
  DROP TABLE IF EXISTS accounts;
  DROP EXTENSION IF EXISTS nram CASCADE;
  DROP ACCESS METHOD IF EXISTS nram;
  DROP FUNCTION IF EXISTS nram_tableam_handler(internal);
  CREATE EXTENSION nram;
  CREATE TABLE accounts (id INT PRIMARY KEY, balance INT) USING nram;
  INSERT INTO accounts VALUES (1, 100), (2, 200), (3, 300);
  SELECT nram_load_policy('2pl');
}

teardown
{
  DROP TABLE accounts;
}

session s1
setup { BEGIN ISOLATION LEVEL SERIALIZABLE; }
step s1_r { SELECT balance FROM accounts WHERE id = 1; }
step s1_w { UPDATE accounts SET balance = balance - 10 WHERE id = 1; }
step s1_c { COMMIT; }

session s2
setup { BEGIN ISOLATION LEVEL SERIALIZABLE; }
step s2_r { SELECT balance FROM accounts WHERE id = 1; }
step s2_w { UPDATE accounts SET balance = balance - 20 WHERE id = 1; }
step s2_c { COMMIT; }

# Under weak control, final balance may be 80 or 90 depending on interleaving.
# Under SERIALIZABLE, one should fail.
permutation s1_w s2_w s1_r s1_c s2_r s2_c
