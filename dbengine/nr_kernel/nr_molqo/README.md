# NeurDB MoLQO - Mixture of Learned Query Optimizer

PostgreSQL extension that sends SQL queries to an external optimizer server and applies optimization hints automatically.

## Installation

```bash
# Compile
sudo -i
cd /code/neurdb-dev/dbengine/nr_kernel/nr_molqo
make clean && make debug && make install

# Configure postgresql.conf
vi /code/neurdb-dev/build/psql/data/postgresql.conf
# Add: shared_preload_libraries = 'pg_hint_plan,nr_molqo'

# Restart PostgreSQL
su - neurdb
/code/neurdb-dev/build/psql/bin/pg_ctl -D /code/neurdb-dev/build/psql/data restart

# Create extension
psql -c "CREATE EXTENSION nr_molqo;"
```

## Test Case 1: SET Command Format

### Without MoLQO (baseline)

```sql
-- Reset and disable MoLQO
RESET ALL;
SET enable_molqo = off;

-- Force Hash Join
SET enable_hashjoin = on;
SET enable_mergejoin = on;
SET enable_nestloop = off;

-- Run query
EXPLAIN ANALYZE
SELECT c.relname, n.nspname
FROM pg_class c
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE c.relkind = 'r';
```

**Result: Hash Join**
```
Hash Join  (cost=1.09..20.62 rows=68 width=128) (actual time=0.053..0.301 rows=72 loops=1)
  Hash Cond: (c.relnamespace = n.oid)
  ->  Seq Scan on pg_class c
  ->  Hash
      ->  Seq Scan on pg_namespace n
Planning Time: 0.316 ms
Execution Time: 0.355 ms
```

### With MoLQO (SET format)

```sql
-- Reset and enable MoLQO
RESET ALL;
SET enable_molqo = on;
SET molqo.server_url = 'http://localhost:8080/optimize_set';

-- Run same query
EXPLAIN ANALYZE
SELECT c.relname, n.nspname
FROM pg_class c
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE c.relkind = 'r';
```

**Result: Nested Loop with Index Scan** (MoLQO changed the plan!)
```
INFO:  ┌─────────────────────────────────────────────────
INFO:  │ MoLQO Optimization Applied
INFO:  │ Server: http://localhost:8080/optimize_set
INFO:  │ Format: SET commands
INFO:  │ Commands: 5 SET statements
INFO:  └─────────────────────────────────────────────────

Nested Loop  (cost=8.51..93.24 rows=68 width=128) (actual time=0.154..0.686 rows=72 loops=1)
  ->  Index Scan using pg_namespace_oid_index on pg_namespace n
  ->  Bitmap Heap Scan on pg_class c
      ->  Bitmap Index Scan on pg_class_relname_nsp_index
Planning Time: 1.974 ms
Execution Time: 0.945 ms
```

## Test Case 2: pg_hint_plan Format

**Note:** Requires `pg_hint_plan` extension installed.
Check via this
```sql
SHOW shared_preload_libraries;
\dx pg_hint_plan
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;
```


```sql
SET enable_molqo = on;
SET molqo.server_url = 'http://localhost:8080/optimize_hint';

EXPLAIN ANALYZE
SELECT c.relname, n.nspname
FROM pg_class c
JOIN pg_namespace n ON c.relnamespace = n.oid
WHERE c.relkind = 'r';
```

**Result: Hash Join with hints**
```
INFO:  ┌─────────────────────────────────────────────────
INFO:  │ MoLQO Optimization Applied
INFO:  │ Server: http://localhost:8080/optimize_hint
INFO:  │ Format: pg_hint_plan
INFO:  │ Hints: /*+ HashJoin(c n) Leading((c n)) BitmapScan(c) BitmapScan(n) */
INFO:  └─────────────────────────────────────────────────

Hash Join  (cost=1.09..20.62 rows=68 width=128) (actual time=0.086..0.397 rows=72 loops=1)
  Hash Cond: (c.relnamespace = n.oid)
  ->  Seq Scan on pg_class c
  ->  Hash
      ->  Seq Scan on pg_namespace n
Planning Time: 2.102 ms
Execution Time: 0.597 ms
```

## Configuration

```sql
-- View current settings
SELECT * FROM molqo_status();

-- Change server URL on the fly
SET molqo.server_url = 'http://localhost:8080/optimize_set';      -- SET format
SET molqo.server_url = 'http://localhost:8080/optimize_hint';     -- Hint format
SET molqo.server_url = 'http://localhost:8080/optimize_join_order'; -- JOIN order

-- Disable MoLQO
SET enable_molqo = off;
```

## Start Test Server

```bash
python3 simple_server.py
```
