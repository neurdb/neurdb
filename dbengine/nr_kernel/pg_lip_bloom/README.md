# pg_lip_bloom

`pg_lip_bloom` is the runtime Bloom-filter extension used by NQO's Filter
action. The Filter implementation invokes exactly these functions:

- `pg_lip_bloom_set_dynamic(integer)`
- `pg_lip_bloom_init(integer)`
- `pg_lip_bloom_add(integer, integer)`
- `pg_lip_bloom_probe(integer, integer)`

The extension currently supports `int4` build and probe keys and at most ten
simultaneous filters. PostgreSQL creates the extension on demand before building
the filters, so the shared library and extension SQL/control files must be
installed before executing a Filter action.

Build and install against the target PostgreSQL installation:

```sh
make PG_CONFIG=/path/to/pg_config
make PG_CONFIG=/path/to/pg_config install
```

Then verify the package in a target database:

```sql
CREATE EXTENSION IF NOT EXISTS pg_lip_bloom;
SELECT pg_lip_bloom_set_dynamic(2);
SELECT pg_lip_bloom_init(1);
```

The old standalone SQL-rewriting prototype is intentionally not part of this
directory. Query rewriting and probe injection now live in PostgreSQL's
`query_split.c` implementation.
