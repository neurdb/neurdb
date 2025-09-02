# occ_rc.spec
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

# =========================
# Sessions
# =========================

session s1
step s1_begin    { BEGIN ISOLATION LEVEL READ COMMITTED; }
step s1_insert   { INSERT INTO rc_test VALUES (1, 'original'); }
step s1_commit   { COMMIT; }

session s2
step s2_begin    { BEGIN ISOLATION LEVEL READ COMMITTED; }
step s2_read    { SELECT * FROM rc_test ORDER BY id; }
step s2_commit   { COMMIT; }

session s3
step s3_begin    { BEGIN ISOLATION LEVEL READ COMMITTED; }
step s3_update   { UPDATE rc_test SET val = 'u1' WHERE id = 1; }
step s3_read { SELECT * FROM rc_test WHERE id = 1; }
step s3_commit   { COMMIT; }

# =========================
# Permutations
# =========================

# P1: should not observe uncommitted insert from peer.
permutation
  s1_begin s1_insert
  s2_begin s2_read
  s1_commit
  s2_commit

# P2: should not observe uncommitted update from peer.
permutation
  s1_begin s1_insert s1_commit
  s2_begin
  s3_begin s3_update s3_read
  s2_read
  s3_commit
  s2_commit

# P3: should observe committed insert and committed update
permutation
  s1_begin s1_insert s1_commit
  s3_begin s3_read s3_update s3_commit
  s2_begin s2_read s2_commit
