import json

import pytest
from experience.collector import ExperienceCollector
from experience.store import ExperienceStore
from optimization.actions import ingest_trajectory


def collector(root, **kwargs):
    worker = ExperienceCollector(
        dataset="job",
        database="imdb_ori",
        policy_log=root / "policy.jsonl",
        db_log=root / "db.jsonl",
        buffer=root / "job.sqlite",
        checkpoint=root / "cursor.json",
        interval=0.01,
        **kwargs,
    )
    worker._load_checkpoint()
    return worker


def append(path, events):
    with path.open("a") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")


def episode(run=1, status="ok", database="imdb_ori"):
    identity = {"pid": 7, "run_id": run, "round": 0}
    states = {
        phase: dict(identity, request_type=phase, **values)
        for phase, values in {
            "dec": {"sql": "SELECT 1"},
            "enum": {"sql": "SELECT 1"},
            "adapt": {"plan_json": {"Node Type": "Result"}},
        }.items()
    }
    actions = {
        "dec": {"dec_action": "skip"},
        "enum": {"enum_action": "native"},
        "adapt": {"filter_action": "none", "ajoin_action": "off"},
    }
    policy = [
        dict(phase="policy_decision", state=state, action=actions[phase])
        for phase, state in states.items()
    ]
    db = [
        dict(
            identity,
            phase="query_start",
            sql="SELECT 1",
            database=database,
            timeout_limit_ms=1000,
            action_settings={},
            ts_ms=1000,
        ),
        dict(
            identity,
            phase="final",
            decision_states=states,
            action={},
            timing_ms={"total": 8},
            ts_ms=1008,
        ),
        dict(
            identity,
            phase="query_complete",
            database=database,
            status=status,
            wall_ms=12,
            runtime_scope="db_nqo",
            result_rows=1,
            ts_ms=1012,
        ),
    ]
    return policy, db


def test_complete_episode_is_durable_and_not_duplicated(tmp_path):
    worker = collector(tmp_path)
    policy, db = episode()
    append(worker.paths["db"], db)
    append(worker.paths["policy"], policy)
    with ExperienceStore(worker.buffer) as store:
        worker.poll_once(store)
        row = next(store.iter_executions())
        assert row["first_runtime_ms"] == 12  # not the final round's 8 ms
        assert len(row["trajectory"]) == 3
        assert row["db_events"][-1]["dataset"] == "job"
        restarted = collector(tmp_path)
        restarted.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 1
        # A lost cursor/replayed log does not overwrite an existing label.
        worker.checkpoint.unlink()
        restarted._load_checkpoint()
        restarted.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 1


def test_pending_episode_survives_restart_and_late_policy(tmp_path):
    worker = collector(tmp_path)
    policy, db = episode()
    append(worker.paths["db"], db[:2])
    with ExperienceStore(worker.buffer) as store:
        worker.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 0
        restarted = collector(tmp_path)
        append(worker.paths["db"], db[2:])
        restarted.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 0
        append(worker.paths["policy"], policy)
        restarted.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 1


def test_partial_line_waits_until_newline(tmp_path):
    worker = collector(tmp_path)
    policy, db = episode()
    append(worker.paths["db"], db)
    append(worker.paths["policy"], policy[:-1])
    with worker.paths["policy"].open("a") as file:
        file.write(json.dumps(policy[-1]))
    with ExperienceStore(worker.buffer) as store:
        worker.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 0
        with worker.paths["policy"].open("a") as file:
            file.write("\n")
        worker.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 1


@pytest.mark.parametrize("status", ["ok", "timeout"])
def test_query_boundaries_are_not_ingested_as_rounds(status):
    policy, db = episode(status=status)
    if status == "timeout":
        db = [db[0], db[-1]]
    result = ingest_trajectory(
        episode_id="test",
        environment_hash="job",
        implementation_version="test",
        db_events=db,
        policy_events=policy,
        timeout_limit_ms=1000,
        episode_status=status,
    )
    assert len(result["trajectory"]) == 3
    assert result["rounds"] == 1
    assert all(d["is_timeout"] == (status == "timeout") for d in result["trajectory"])


@pytest.mark.parametrize("status", ["timeout", "error", "cancelled"])
def test_failed_execution_and_active_decisions_are_retained(tmp_path, status):
    worker = collector(tmp_path)
    policy, db = episode(status=status)
    append(worker.paths["db"], [db[0], db[-1]])
    append(worker.paths["policy"], policy)
    with ExperienceStore(worker.buffer) as store:
        worker.poll_once(store)
        row = next(store.iter_executions())
        assert row["status"] == status
        assert len(row["db_events"][-1]["policy_events"]) == 3
        if status == "timeout":
            assert len(row["trajectory"]) == 3
            assert all(d["is_timeout"] for d in row["trajectory"])


def test_database_filter_and_multiple_sqls_share_logs(tmp_path):
    worker = collector(tmp_path)
    for run, database in [(1, "imdb_ori"), (2, "imdb_ori"), (3, "tpch")]:
        policy, db = episode(run, database=database)
        append(worker.paths["db"], db)
        append(worker.paths["policy"], policy)
    with ExperienceStore(worker.buffer) as store:
        worker.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 2
        assert worker.state["pending"] == {}


def test_bootstrap_preserved_and_only_one_writer(tmp_path):
    seed = tmp_path / "seed.sqlite"
    with ExperienceStore(seed):
        pass
    original = seed.read_bytes()
    worker = collector(tmp_path, bootstrap=seed)
    worker.start()
    try:
        assert worker.ready.wait(3)
        assert worker.startup_error is None
        duplicate = collector(tmp_path)
        duplicate.start()
        assert duplicate.ready.wait(3)
        duplicate.stop()
        assert duplicate.startup_error is not None
        assert seed.read_bytes() == original
    finally:
        worker.stop()
    assert not worker.is_alive()


def test_checkpoint_failure_replays_idempotently(tmp_path, monkeypatch):
    worker = collector(tmp_path)
    policy, db = episode()
    append(worker.paths["db"], db)
    append(worker.paths["policy"], policy)
    with ExperienceStore(worker.buffer) as store:

        def fail_checkpoint():
            raise OSError("disk full")

        with monkeypatch.context() as patch:
            patch.setattr(worker, "_save_checkpoint", fail_checkpoint)
            with pytest.raises(OSError):
                worker.poll_once(store)
        worker._load_checkpoint()
        worker.poll_once(store)
        assert store.trajectory_cache_summary()["entries"] == 1


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval(tmp_path, interval):
    with pytest.raises(ValueError):
        ExperienceCollector(
            dataset="job",
            database="imdb_ori",
            policy_log=tmp_path / "p",
            db_log=tmp_path / "d",
            buffer=tmp_path / "b",
            checkpoint=tmp_path / "c",
            interval=interval,
        )
