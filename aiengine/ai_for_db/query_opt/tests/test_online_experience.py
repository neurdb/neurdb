from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
from experience.store import (
    REPLAY_CACHE_COLUMNS,
    ExperienceStore,
    decode_payload,
    encode_payload,
    semantic_trajectory_hash,
)


def execution_payload(
    *,
    query_id: str = "1a",
    source_episode_id: str = "episode-1",
    runtime_source: str = "physical",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "sql_hash": f"sql-{query_id}",
        "trajectory": [
            {
                "phase": "high",
                "state": {"request_type": "dec", "relations": 4},
                "state_hash": "state-high",
                "action": {"dec_action": "stop"},
                "runtime_source": runtime_source,
                "policy": {
                    "inference_mode": "stochastic",
                    "action_index": 0,
                    "log_probability": -0.25,
                    "predicted_value": -1.5,
                    "action_mask": [1.0, 1.0],
                },
            }
        ],
        "db_events": [
            {
                "phase": "final",
                "round": 0,
                "timing_ms": {"total": 12.0},
            }
        ],
        "status": "ok",
        "first_runtime_ms": 12.0,
        "charged_runtime_ms": 12.0,
        "timeout_limit_ms": 60_000,
        "action_config_hash": "config-v1",
        "result_hash": "result",
        "result_rows": 1,
        "source_episode_id": source_episode_id,
    }


def test_store_creates_only_canonical_replay_cache_table() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path) as store:
            tables = {
                row[0]
                for row in store.db.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            columns = {
                row[1] for row in store.db.execute("PRAGMA table_info(replay_cache)")
            }
        assert tables == {"replay_cache"}
        assert columns == set(REPLAY_CACHE_COLUMNS)


def test_execution_and_policy_training_payload_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path) as store:
            cache_id, inserted = store.append_execution(**execution_payload())
            assert inserted
            rows = list(store.iter_executions())
            assert len(rows) == 1
            assert rows[0]["cache_id"] == cache_id
            assert rows[0]["trajectory"][0]["policy"]["log_probability"] == -0.25
            assert rows[0]["db_events"][0]["timing_ms"]["total"] == 12.0
            assert store.training_summary() == {
                "episodes": 1,
                "timeouts": 0,
                "wrong_results": 0,
                "measured_wall_ms": 12.0,
                "decisions": 1,
                "unique_subquery_action_pairs": 1,
            }


def test_identical_episode_is_idempotent_but_distinct_measurements_are_retained() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path) as store:
            first, first_inserted = store.append_execution(**execution_payload())
            second, second_inserted = store.append_execution(**execution_payload())
            third, third_inserted = store.append_execution(
                **execution_payload(source_episode_id="episode-2")
            )
            assert first_inserted
            assert not second_inserted
            assert first == second
            assert third_inserted
            assert third != first
            assert len(list(store.iter_executions())) == 2


def test_merge_is_one_table_and_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source_path = root / "source_light.sql"
        output_path = root / "output_light.sql"
        with ExperienceStore(source_path) as source:
            source.append_execution(**execution_payload())
        with ExperienceStore(output_path) as output:
            assert output.merge_from(source_path) == {"replay_cache": 1}
            assert output.merge_from(source_path) == {"replay_cache": 0}
            assert len(list(output.iter_executions())) == 1


def test_cache_replay_measurements_are_not_reused_as_physical_candidates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path) as store:
            store.append_execution(
                **execution_payload(
                    source_episode_id="cached-episode",
                    runtime_source="cache",
                )
            )
            assert store.trajectory_cache_candidates(sql_hash="sql-1a") == []


def test_read_only_store_rejects_writes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path):
            pass
        with ExperienceStore(path, read_only=True) as store:
            with pytest.raises(RuntimeError, match="read-only"):
                store.append_execution(**execution_payload())


def test_legacy_lightweight_record_is_canonicalized_only_in_memory() -> None:
    legacy = [
        {
            "phase": "high",
            "state": {"request_type": "high", "relations": 4},
            "state_hash": "legacy-state",
            "action": {"high_action": "split"},
        }
    ]
    legacy_hash = semantic_trajectory_hash(legacy)
    encoding, payload, blob_hash = encode_payload(legacy)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experience_light.sql"
        with ExperienceStore(path) as store:
            cache_id, _ = store.append_execution(**execution_payload())
            store.db.execute(
                "UPDATE replay_cache SET trajectory_hash=?, "
                "trajectory_blob_hash=?, trajectory_encoding=?, "
                "trajectory_payload=? WHERE cache_id=?",
                (legacy_hash, blob_hash, encoding, payload, cache_id),
            )
            store.db.commit()

        with ExperienceStore(path, read_only=True) as store:
            item = next(store.iter_executions())
            decision = item["trajectory"][0]
            assert decision["phase"] == "dec"
            assert decision["state"]["request_type"] == "dec"
            assert decision["action"]["dec_action"] == "apply"
            assert "high_action" not in decision["action"]
            assert decision["state_hash"] != "legacy-state"
            assert item["stored_trajectory_hash"] == legacy_hash
            assert item["canonical_trajectory_hash"] == semantic_trajectory_hash(
                item["trajectory"]
            )
            assert item["canonical_trajectory_hash"] != legacy_hash

        merged_path = Path(tmp) / "merged_light.sql"
        with ExperienceStore(merged_path) as merged:
            assert merged.merge_from(path) == {"replay_cache": 1}
            raw = merged.db.execute(
                "SELECT trajectory_encoding, trajectory_payload " "FROM replay_cache"
            ).fetchone()
            persisted = decode_payload(str(raw[0]), raw[1])
            assert persisted[0]["phase"] == "dec"
            assert persisted[0]["action"]["dec_action"] == "apply"
            assert "high_action" not in persisted[0]["action"]


def test_legacy_multitable_database_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY)")
        db.commit()
        db.close()
        with pytest.raises(RuntimeError, match="expected lightweight replay_cache"):
            ExperienceStore(path)


def test_semantic_trajectory_hash_ignores_annotations_and_raw_state_payload() -> None:
    first = [
        {
            "phase": "high",
            "state": {"backend_pid": 1},
            "state_hash": "stable-state",
            "action": {"dec_action": "split"},
            "policy": {"log_probability": -0.1},
        }
    ]
    second = [
        {
            "phase": "high",
            "state": {"backend_pid": 2},
            "state_hash": "stable-state",
            "action": {"dec_action": "split"},
            "policy": {"log_probability": -0.9},
        }
    ]
    assert semantic_trajectory_hash(first) == semantic_trajectory_hash(second)
