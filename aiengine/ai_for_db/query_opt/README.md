# NeurDB Query Optimization

NQO is the AI-for-DB query optimization service. It runs separately from the
analytics server in `aiengine/runtime`; it does not consume streamed table
batches or replace `aiengine/network`.

## Layout and Execution

```text
query_opt/
  src/runtime/        HTTP action server and fixed/replay policies
  src/model/          graph/plan encoders and hierarchical policy
  src/optimization/   action vocabulary, state contracts, dataset parameters
  src/database/       PostgreSQL catalog reader and SQL client
  src/experience/     persistent SQLite experience storage
  src/training/       explicit training, mixed workloads, checkpoint updates
  examples/           SQL and Python clients
  tests/              policy, experience, training, and client regression tests
```

SQL goes to PostgreSQL. After parse/analyze/rewrite, the backend calls the
server sequentially: **Dec** on the residual query graph; **Sched** to select a
candidate if decomposing; **Enum** on the selected query graph; **Adapt** on the
selected physical plan. PostgreSQL then executes, materializes intermediate
results as needed, and rewrites the residual query for the next round. Dec/Enum
do not consume a baseline plan tree; Adapt does not consume the query graph.

Kernel ownership, relative to the NeurDB root:

- `dbengine/src/backend/parser/query_split.c`: rounds, state serialization,
  decomposition/scheduling, DP top-K + Leading hints, and SPI Bloom builds.
- `dbengine/src/backend/executor/nodeNqoAdaptiveJoin.c` and `nodeHashjoin.c`:
  execution-time adaptive join behavior.
- `dbengine/src/backend/tcop/postgres.c`, `utils/misc/guc_tables.c`, and matching
  headers: integration and session settings.
- `dbengine/nr_kernel/pg_lip_bloom`: Filter action's Bloom extension.

This package imports the existing NQO runtime from the `neurqo` repository at
`dd44135f115ac41e5c2e67c210ffe9e21ee8dae3`, retaining module names and checkpoint
compatibility. Workloads, benchmark results, large buffers, and checkpoints are
not duplicated here. The original research repository remains the experiment
and reproduction workspace. Synchronize future algorithm changes explicitly.

## Install

Use a dedicated **Python 3.11** environment. Do not install into the analytics
server's environment: it has different NumPy/PyTorch requirements.

Inside the development container:

```bash
cd /code/neurdb-dev/aiengine/ai_for_db/query_opt
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
PATH=/usr/local/bin:/usr/bin:/bin uv pip install --python .venv/bin/python -e '.[test]'
source .venv/bin/activate
export PGHOST=127.0.0.1 PGPORT=5432 PGUSER=neurdb PGDATABASE=imdb_ori
```

Use libpq environment variables or `.pgpass` for credentials. The existing
container is `neurdb_dev_opt`; its PostgreSQL port is `5432` internally and
`15432` on the host. `/home/naili/neurdb` is mounted at `/code/neurdb-dev`.
Other deployments should substitute their own checkout and connection paths.
Building `psycopg2` requires a C compiler and standard PostgreSQL client headers
(`libpq-dev` on Ubuntu). Use the system `pg_config` for that Python driver:
NeurDB's custom `0.2-devel` version is rejected by psycopg2's version check.
This does not change the server to which the driver connects. In contrast,
build server extensions against **NeurDB's** `pg_config`.

The backend must include the NQO kernel changes and load `pg_hint_plan` through
`shared_preload_libraries`. Build/install `dbengine/nr_kernel/pg_lip_bloom` with
the target `PG_CONFIG` (also included in `nr_kernel`'s build/install targets).
Provision `CREATE EXTENSION pg_lip_bloom` as an administrator before using LIP
with a restricted database role. Existing development databases already have
the required runtime components.

## Run

Fixed-policy connectivity test, in the foreground:

```bash
NQO_FIXED_DEC=apply NQO_FIXED_SCHED_ALPHA=0.5 \
  nqo-server --model-module runtime.policies.fixed:predict --require-model
```

In another shell using the same environment:

```bash
nqo-sql --dataset job --file examples/query.sql
nqo-sql --pg --file examples/query.sql
python examples/query.py --dataset job --sql 'SELECT MIN(title) FROM title'
```

The client opens a fresh session, applies only session-local settings, executes
once, and closes the session (including temporary tables). `wall_ms` covers
`execute` through fetching the result, including DB planning, AI round trips,
and execution, but excludes connection setup and session configuration. For a
multi-statement file it covers the whole batch and returns only the final
statement's result. Files must contain SQL, not psql meta-commands. The default
`statement_timeout` is 60 seconds per statement; errors/timeouts exit nonzero.

`--dataset` loads the copied dataset-level action parameters; it does not pick
the actions or override learned scheduling. Fixed-policy testing uses
`NQO_FIXED_ENUM=top5`, `NQO_FIXED_FILTER=selective`, and/or
`NQO_FIXED_AJOIN=conservative` on the server. For TPC-H, keep Dec at `skip`.
Standalone QuerySplit uses alpha 0.5; learned scheduling is not fixed unless
explicitly requested.

To use an existing checkpoint:

```bash
nqo-sql --export-catalog .runtime/job-catalog.json
nqo-server --model-path /path/to/trusted/best.pt \
  --catalog-path .runtime/job-catalog.json --workload job --device cpu \
  --require-model --trajectory-log .runtime/policy.jsonl
```

Stop the fixed server before starting a model server on the same port. Use
`--model-hidden` if the checkpoint's hidden size differs from 128. Catalogs are
read at startup; PG sends current residual/temporary-relation context per round.
Checkpoints must be trusted because loading includes PyTorch pickle metadata.
`--require-model` fails startup instead of silently using the stub policy.
Clear fixed-policy environment variables before learned inference, especially
`NQO_FIXED_SCHED_ALPHA`.

The server accepts JSON state via `POST /action` and returns the existing
line-oriented PG wire format. `GET /` reports health/source; `POST /reload`
explicitly replaces the loaded policy. It defaults to loopback and has no
authentication: keep the control port on a trusted private network, never expose
it directly to the Internet. For a separate AI host, change the client's
`--server-url` to an address reachable from the PostgreSQL container.

## Experience and Training

`experience.store.ExperienceStore` preserves the dataset-level SQLite store and
complete-trajectory reuse semantics. The server's `--trajectory-log` records
policy decisions; PG's `nqo.trajectory_log` records execution events. They are
distinct logs, not automatically a populated training buffer. The research
collector still combines decisions, execution events, runtimes, and correctness
checks into experience records. This SQL example is not an experiment collector.

`nqo-train`, `nqo-train-mixed`, and `nqo-incremental-trainer` expose the existing
training/checkpoint tools (`--help` for arguments). Nothing starts training,
collection, scheduled jobs, or background services automatically. Keep local
catalogs, logs, buffers, and checkpoints under ignored `.runtime/`.

## Verify

```bash
pytest -q
```

Tests use synthetic fixtures and temporary SQLite files, not workload training.
Opt-in integration tests execute JOB 2a against `PGHOST/PGPORT/PGUSER/PGDATABASE`,
compare fixed-action and checkpoint results with PG, and check actual action
activation and state boundaries. They launch temporary loopback action servers
and always stop them on exit; they do not train or run a workload benchmark.

```bash
NQO_TEST_DB=1 NQO_TEST_MODEL=/path/to/trusted/best.pt \
  NQO_TEST_CATALOG="$PWD/.runtime/job-catalog.json" \
  pytest -q tests/test_database_integration.py
```

Omit the model/catalog variables to run only the fixed-policy tests. The pytest
temporary directory must be writable by the PostgreSQL OS user for execution
logs; in `neurdb_dev_opt`, run tests as the default `neurdb` user.
