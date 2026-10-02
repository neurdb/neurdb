# NeurDB Query Optimization

The system name is **NeurQO**; new commands use `neurqo-*`, PostgreSQL settings
use `neurqo.*`, and environment variables use `NEURQO_*`. Existing `NQO_*`
environment variables and `nqo.*` GUCs remain read aliases. Old checkpoints and
buffers are loaded without rewriting their contents; profile readers and cache
hashes retain compatibility with the released `nqo_enabled` field and labels.

NeurQO is the AI-for-DB query optimization service. It runs separately from the
analytics server in `aiengine/runtime`; it does not consume streamed table
batches or replace `aiengine/network`.

## Layout and Execution

```text
query_opt/
  src/runtime/        HTTP action server and fixed/replay policies
  src/model/          graph/plan encoders and hierarchical policy
  src/optimization/   action vocabulary, state contracts, dataset parameters
  src/database/       PostgreSQL catalog reader and SQL client
  src/experience/     persistent SQLite storage and background log collector
  src/training/       explicit training, mixed workloads, checkpoint updates
  data/bootstrap/     versioned initial experience, including JOB
  data/experience/    writable dataset buffers (ignored by Git)
  data/logs/          policy and DB JSONL files (ignored by Git)
  data/collector/     durable log read positions (ignored by Git)
  examples/           SQL/Python clients
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
- `dbengine/src/backend/executor/nodeNeurQOAdaptiveJoin.c` and `nodeHashjoin.c`:
  execution-time adaptive join behavior.
- `dbengine/src/backend/tcop/postgres.c`, `utils/misc/guc_tables.c`, and matching
  headers: integration and session settings.
- `dbengine/nr_kernel/pg_lip_bloom`: Filter action's Bloom extension.

This package imports the existing NeurQO runtime from the `neurqo` repository at
`dd44135f115ac41e5c2e67c210ffe9e21ee8dae3`, retaining module names and checkpoint
compatibility. The released lightweight JOB buffer is included as
[bootstrap data](data/bootstrap/README.md); full collection buffers, benchmark
results, workloads, and checkpoints are not duplicated here. The original
research repository remains the experiment and reproduction workspace.
Synchronize future algorithm changes explicitly.

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

The backend must include the NeurQO kernel changes and load `pg_hint_plan` through
`shared_preload_libraries`. Build/install `dbengine/nr_kernel/pg_lip_bloom` with
the target `PG_CONFIG` (also included in `nr_kernel`'s build/install targets).
Provision `CREATE EXTENSION pg_lip_bloom` as an administrator before using LIP
with a restricted database role. Existing development databases already have
the required runtime components.

## Run

Fixed-policy connectivity test, in the foreground:

```bash
NEURQO_FIXED_DEC=apply NEURQO_FIXED_SCHED_ALPHA=0.5 \
  neurqo-server --model-module runtime.policies.fixed:predict --require-model
```

In another shell using the same environment:

```bash
neurqo-sql --dataset job --file examples/query.sql
neurqo-sql --pg --file examples/query.sql
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
`NEURQO_FIXED_ENUM=top5`, `NEURQO_FIXED_FILTER=selective`, and/or
`NEURQO_FIXED_AJOIN=conservative` on the server. For TPC-H, keep Dec at `skip`.
Standalone QuerySplit uses alpha 0.5; learned scheduling is not fixed unless
explicitly requested.

To use an existing checkpoint:

```bash
neurqo-sql --export-catalog .runtime/job-catalog.json
neurqo-server --model-path /path/to/trusted/best.pt \
  --catalog-path .runtime/job-catalog.json --workload job --device cpu \
  --require-model --trajectory-log .runtime/policy.jsonl
```

Stop the fixed server before starting a model server on the same port. Use
`--model-hidden` if the checkpoint's hidden size differs from 128. Catalogs are
read at startup; PG sends current residual/temporary-relation context per round.
Checkpoints must be trusted because loading includes PyTorch pickle metadata.
`--require-model` fails startup instead of silently using the stub policy.
Clear fixed-policy environment variables before learned inference, especially
`NEURQO_FIXED_SCHED_ALPHA`.

The server accepts JSON state via `POST /action` and returns the existing
line-oriented PG wire format. `GET /` reports health/source; `POST /reload`
explicitly replaces the loaded policy. It defaults to loopback and has no
authentication: keep the control port on a trusted private network, never expose
it directly to the Internet. For a separate AI host, change the client's
`--server-url` to an address reachable from the PostgreSQL container.

## Experience and Training

`experience.store.ExperienceStore` preserves the dataset-level SQLite store and
complete-trajectory reuse semantics. Enable the server's collector explicitly:

```bash
neurqo-server --model-module runtime.policies.fixed:predict --require-model \
  --workload job --collect-experience --experience-database imdb_ori
```

This starts one server-owned thread, polling every two seconds. It combines
policy decisions and DB execution events, then appends completed executions to
`data/experience/job.sqlite`. `--data-dir` (or `NEURQO_DATA_DIR`) changes the data
root; `--collector-interval` changes the polling interval. Source installations
default to this package's `data/`; installed wheels without a checkout use
`~/.local/share/neurdb/query_opt`. No training starts.

Configure PostgreSQL to write the corresponding DB log, using an absolute path
writable by its OS user. For the development container, an administrator can
set this default for **new connections** once:

```sql
ALTER DATABASE imdb_ori SET neurqo.trajectory_log =
  '/code/neurdb-dev/aiengine/ai_for_db/query_opt/data/logs/job.db.jsonl';
```

Then execute NeurQO-enabled SQL normally, for example `neurqo-sql --dataset job --file
examples/query.sql`. psql applications need the usual `SET neurqo = on` and correct
`neurqo.server_url`; they do not manage SQLite. Without a database default, use a
session-local `SET neurqo.trajectory_log` or the client's `--trajectory-log`.
The collector's `--db-trajectory-log` must identify that same file. On separate
hosts, expose the PG-written log to the server through a shared mount; the paths
on the two hosts need not be identical. The server cannot configure remote PG
filesystem permissions or database defaults for you.

### Persistent Files

Each dataset uses **two append-only JSONL logs shared by all queries**, not a
new file per SQL. Each line is one JSON event:

| File under `data/` | Writer | Contents |
|---|---|---|
| `logs/job.policy.jsonl` | AI server | `policy_decision`: state, action, inference latency and policy metadata; optional reload events |
| `logs/job.db.jsonl` | PostgreSQL | `query_start`, per-round `split`/`final`, and `query_complete` |
| `experience/job.sqlite` | Collector | One persistent execution record with its full decision/execution trajectory |
| `collector/job.json` | Collector | Read offsets and pending executions for restart recovery, not a SQL log |

`pid` and `run_id` correlate a SQL execution across both logs; `round` and
`request_type` associate individual decisions. Start events include the original
statement, database, timeout and action settings. Round events retain states,
selected actions, timing breakdowns, adaptive-join counters and materialized
relation statistics. Completion events record status (`ok`, `timeout`, `error`,
or `cancelled`), elapsed time, result row count when available, and error details.
Concurrent writes are serialized so JSON lines do not interleave. Logs contain
SQL text and plan/state details: restrict access as for database query logs.

The collector resumes saved offsets after restart, waits for complete JSON
lines, and uses stable execution IDs to avoid duplicate inserts on replay.
SQLite writes commit before advancing the cursor. It only accepts the configured
database and permits one collector per runtime buffer; changing datasets requires
a server restart. `GET /` reports collector health. Do not truncate or rotate logs
with unread events: drain them first. Abrupt backend termination without a
completion event leaves a pending execution, never a fabricated success.

The tracked [JOB bootstrap](data/bootstrap/README.md) has 11,045 historical
records covering 113 queries. On first collection startup it is copied to
`data/experience/job.sqlite` **only if that runtime file does not exist**. All
subsequent appends go to the runtime file; the tracked bootstrap is unchanged.
Datasets without bootstrap data start with an empty store. Runtime SQLite files,
their lock files, logs and cursors are ignored by Git.

### Measurement Scope

New service-collected records use `runtime_scope=db_neurqo`: elapsed time from entry
to exit of the NeurQO hook, covering all rounds, search/planning, AI calls, execution
and intermediate materialization. This is **not just the final SELECT** and is
not identical to client `wall_ms`: parse/analyze/rewrite, connection setup and
client/network fetch overhead are outside this timer. The scope and action
settings are included in the stored configuration identity, separate from
historical experiment measurements. Timeout records retain the observed elapsed
time; a benchmark-specific timeout penalty needs a PG baseline and is applied by
evaluation, not guessed by this collector.

Collection records execution facts, not a correctness comparison against PG;
it does not fetch/hash result rows or execute SQL again. Statements that never
enter the NeurQO hook (NeurQO disabled, non-SELECT, parse/analyze errors, or the currently
unsupported extended-protocol path) do not generate these experiences. Experience
is not a result cache: ordinary SQL is still executed, and intermediate result
tables cannot be restored from it. Training/replay consumers decide how to reuse
labels, keeping dataset and measurement scope consistent.

`neurqo-train`, `neurqo-train-mixed`, and `neurqo-incremental-trainer` expose the existing
training/checkpoint tools (`--help` for arguments). Without `--collect-experience`
the server does not start a collector. Nothing installs scheduled services or
starts training automatically. Keep local catalogs and checkpoints in ignored
`.runtime/` and persistent collection data in `data/`.

## Verify

```bash
pytest -q
```

Tests use synthetic fixtures and temporary SQLite files, not workload training.
Opt-in integration tests execute JOB 2a against `PGHOST/PGPORT/PGUSER/PGDATABASE`,
compare fixed-action and checkpoint results with PG, and check actual action
activation and state boundaries. They launch temporary loopback action servers
and always stop them on exit; they do not train or run a workload benchmark.
The collector smoke test also checks a split execution, timeout, execution error,
multiple statements sharing logs, and restart deduplication.

```bash
NEURQO_TEST_DB=1 NEURQO_TEST_MODEL=/path/to/trusted/best.pt \
  NEURQO_TEST_CATALOG="$PWD/.runtime/job-catalog.json" \
  pytest -q tests/test_database_integration.py
```

Omit the model/catalog variables to run only the fixed-policy tests. The pytest
temporary directory must be writable by the PostgreSQL OS user for execution
logs; in `neurdb_dev_opt`, run tests as the default `neurdb` user.
