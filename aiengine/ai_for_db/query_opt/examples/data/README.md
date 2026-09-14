# JOB Experience Sample

`job_light.sqlite` is the released lightweight JOB experience buffer, copied
byte-for-byte from `neurqo/results/buffers/job_light.sql` at source commit
`dd44135f115ac41e5c2e67c210ffe9e21ee8dae3`. The source's historical `.sql` suffix
also denotes a binary SQLite database, not a SQL script. Only the filename has
changed; there is no new data collection or relabeling.

| Property | Value |
|---|---|
| Dataset | JOB / IMDb |
| Size | 83,267,584 bytes (79.4 MiB) |
| Table | `replay_cache` |
| Executions | 11,045 |
| Distinct query IDs | 113 |
| Status counts | 10,844 `ok`, 201 `timeout` |

SHA-256:
`1b14b70d16dfc3db188fdaa40b491201fd223d28bb2e082680e0632220cd97e6`

Each record stores SQL/action-configuration identities, the observed and charged
runtimes, timeout/status metadata, optional result hashes, and compressed
policy/execution trajectories. Some records, including PG baseline executions,
have empty trajectories. The store reader normalizes legacy action names in
memory without rewriting this released snapshot.

## Inspect

From `aiengine/ai_for_db/query_opt`, with its Python environment activated:

```bash
python - <<'PY'
from experience.store import ExperienceStore

with ExperienceStore("examples/data/job_light.sqlite", read_only=True) as store:
    print(store.trajectory_cache_summary())
    record = next(
        row for row in store.iter_executions(query_ids=["2a"])
        if row["trajectory"]
    )
    print("query:", record["query_id"], "status:", record["status"])
    print("end-to-end ms:", record["first_runtime_ms"])
    print("charged ms:", record["charged_runtime_ms"])
    print("decisions:", len(record["trajectory"]))
    print("DB events:", len(record["db_events"]))
PY
```

This inspection does not connect to PostgreSQL, run SQL, or start training.
The existing training tool accepts this path through `--experience-db`; run
`nqo-train --help` for the required catalog, query selection, and model options.
Use a catalog and workload definition matching the JOB database. Historical
timings are measurements from the original experiment environment, not a fresh
performance measurement on the current machine.

## Runtime Use

Treat this tracked file as a **read-only seed**. For a writable local copy:

```bash
mkdir -p .runtime/experience
cp -n examples/data/job_light.sqlite .runtime/experience/job.sqlite
```

Open the local copy with `ExperienceStore` and explicitly call
`append_execution()` with collected states/actions, DB events, execution status,
and runtimes. `.runtime/` is ignored by Git; do not write new runs into the
tracked example. The copy command leaves an existing local buffer untouched.

**Executing SQL does not currently append experience automatically.** Neither
psql nor `nqo-sql` loads this file implicitly; the AI server logs policy
decisions but does not yet collect complete SQL outcomes into SQLite. Production
collection should be connected on the service/DB side so applications need not
manage the buffer. That is a separate integration task, not enabled by shipping
this sample. Experience also does not contain the materialized SQL result tables
needed to resume a partially executed query.
