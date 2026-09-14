#!/usr/bin/env python3
"""Single-table execution experience used by every NQO workflow."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from optimization.action_vocabulary import (
    canonical_phase,
    normalize_policy_action,
    normalize_policy_state,
)

REPLAY_CACHE_COLUMNS = (
    "cache_id",
    "query_id",
    "sql_hash",
    "trajectory_hash",
    "created_at_ms",
    "action_config_hash",
    "status",
    "first_runtime_ms",
    "charged_runtime_ms",
    "timeout_limit_ms",
    "result_hash",
    "result_rows",
    "source_episode_id",
    "trajectory_blob_hash",
    "trajectory_encoding",
    "trajectory_payload",
    "db_events_blob_hash",
    "db_events_encoding",
    "db_events_payload",
    "bindings_json",
)

REPLAY_CACHE_REQUIRED_COLUMNS = frozenset(REPLAY_CACHE_COLUMNS[:-1])

REPLAY_CACHE_SCHEMA = """
CREATE TABLE replay_cache (
    cache_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    sql_hash TEXT NOT NULL,
    trajectory_hash TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    action_config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    first_runtime_ms REAL NOT NULL,
    charged_runtime_ms REAL NOT NULL,
    timeout_limit_ms INTEGER NOT NULL,
    result_hash TEXT,
    result_rows INTEGER,
    source_episode_id TEXT NOT NULL,
    trajectory_blob_hash TEXT NOT NULL,
    trajectory_encoding TEXT NOT NULL,
    trajectory_payload BLOB NOT NULL,
    db_events_blob_hash TEXT NOT NULL,
    db_events_encoding TEXT NOT NULL,
    db_events_payload BLOB NOT NULL,
    bindings_json TEXT
);
CREATE INDEX replay_cache_sql_idx
ON replay_cache(sql_hash, created_at_ms, cache_id);
CREATE UNIQUE INDEX replay_cache_config_idx
ON replay_cache(sql_hash, action_config_hash, cache_id);
"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = canonical_json(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def semantic_trajectory_hash(trajectory: Iterable[dict[str, Any]]) -> str:
    """Hash only stable state/action identity, not training annotations."""
    return content_hash(
        [
            {
                "phase": item["phase"],
                "state_hash": item["state_hash"],
                "action": item["action"],
            }
            for item in trajectory
        ]
    )


def canonical_trajectory(
    trajectory: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize legacy records in memory; callers never mutate old buffers."""
    # Imported lazily to avoid the module-level actions -> store dependency.
    from optimization.actions import stable_state

    normalized: list[dict[str, Any]] = []
    for stored_decision in trajectory:
        decision = dict(stored_decision)
        phase = canonical_phase(decision.get("phase"))
        decision["phase"] = phase
        decision["state"] = normalize_policy_state(decision.get("state") or {})
        decision["state_hash"] = content_hash(stable_state(decision["state"]))
        decision["action"] = normalize_policy_action(
            decision.get("action") or {}, phase=phase
        )
        decision["policy"] = normalize_policy_action(
            decision.get("policy") or decision.get("action") or {}, phase=phase
        )
        normalized.append(decision)
    return normalized


def encode_payload(value: Any) -> tuple[str, bytes, str]:
    raw = canonical_json(value).encode("utf-8")
    return "zlib", zlib.compress(raw, level=6), hashlib.sha256(raw).hexdigest()


def decode_payload(encoding: str, payload: bytes) -> Any:
    raw = bytes(payload)
    if encoding == "zlib":
        raw = zlib.decompress(raw)
    elif encoding not in {"json", "plain", "raw"}:
        raise RuntimeError(f"unsupported replay-cache encoding: {encoding}")
    return json.loads(raw.decode("utf-8"))


class ExperienceStore:
    """Read and write the lightweight one-table ``replay_cache`` format.

    The buffer contains execution experience only. Experiment manifests,
    baselines, fold assignments, warmups, and checkpoints live outside it.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        writable: Optional[bool] = None,
    ) -> None:
        if writable is not None:
            read_only = not writable
        self.path = Path(path).resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            uri = f"{self.path.as_uri()}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, timeout=30.0)
            self.db.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(self.path), timeout=30.0)
            self.db.execute("PRAGMA busy_timeout=30000")
            self.db.execute("PRAGMA synchronous=FULL")
        self.db.row_factory = sqlite3.Row
        if not self.read_only:
            self._create_schema_if_empty()
        self._validate_schema()
        self.action_config_hash: str | None = None
        self.binding: dict[str, str] | None = None

    def _create_schema_if_empty(self) -> None:
        if not self._tables():
            self.db.executescript(REPLAY_CACHE_SCHEMA)
            self.db.commit()

    def _tables(self) -> set[str]:
        return {
            str(row[0])
            for row in self.db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    def _validate_schema(self) -> None:
        tables = self._tables()
        if tables != {"replay_cache"}:
            raise RuntimeError(
                f"expected lightweight replay_cache in {self.path}, "
                f"found tables {sorted(tables)}"
            )
        self.columns = {
            str(row[1]) for row in self.db.execute("PRAGMA table_info(replay_cache)")
        }
        missing = REPLAY_CACHE_REQUIRED_COLUMNS - self.columns
        if missing:
            raise RuntimeError(
                f"lightweight buffer {self.path} is missing columns "
                f"{sorted(missing)}"
            )
        self.has_bindings = "bindings_json" in self.columns

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @staticmethod
    def now_ms() -> int:
        return time.time_ns() // 1_000_000

    def set_action_config_hash(self, value: str) -> None:
        self.action_config_hash = str(value)

    def set_binding(self, *, protocol: str, fold: str, checkpoint_sha256: str) -> None:
        self.binding = {
            "protocol": protocol,
            "fold": fold,
            "checkpoint_sha256": checkpoint_sha256,
        }

    def append_execution(
        self,
        *,
        query_id: str,
        sql_hash: str,
        trajectory: list[dict[str, Any]],
        db_events: list[dict[str, Any]],
        status: str | None = None,
        first_runtime_ms: float | None = None,
        charged_runtime_ms: float | None = None,
        timeout_limit_ms: int | None = None,
        action_config_hash: str | None = None,
        result_hash: str | None = None,
        result_rows: int | None = None,
        source_episode_id: str | None = None,
        cache_id: str | None = None,
        created_at_ms: int | None = None,
        bindings: list[dict[str, str]] | None = None,
    ) -> tuple[str, bool]:
        if self.read_only:
            raise RuntimeError(f"buffer is read-only: {self.path}")
        if status is None or first_runtime_ms is None or charged_runtime_ms is None:
            raise ValueError("execution status and runtimes are required")
        if timeout_limit_ms is None:
            raise ValueError("timeout_limit_ms is required")
        action_config_hash = str(
            action_config_hash or self.action_config_hash or "default"
        )
        # Accept legacy caller objects during the transition, but persist only
        # the canonical paper-aligned protocol in newly created buffers.
        trajectory = canonical_trajectory(trajectory)
        trajectory_hash = semantic_trajectory_hash(trajectory)
        if source_episode_id is None:
            source_episode_id = (
                "trajectory_"
                + content_hash(
                    {
                        "sql_hash": sql_hash,
                        "trajectory_hash": trajectory_hash,
                        "action_config_hash": action_config_hash,
                    }
                )[:24]
            )
        cache_id = cache_id or content_hash(
            {
                "source_episode_id": source_episode_id,
                "sql_hash": sql_hash,
                "trajectory_hash": trajectory_hash,
                "action_config_hash": action_config_hash,
            }
        )
        trajectory_encoding, trajectory_payload, trajectory_blob_hash = encode_payload(
            trajectory
        )
        db_encoding, db_payload, db_blob_hash = encode_payload(db_events)
        columns = list(REPLAY_CACHE_COLUMNS[:-1])
        values: list[Any] = [
            cache_id,
            str(query_id),
            str(sql_hash),
            trajectory_hash,
            int(self.now_ms() if created_at_ms is None else created_at_ms),
            str(action_config_hash),
            str(status),
            float(first_runtime_ms),
            float(charged_runtime_ms),
            int(timeout_limit_ms),
            result_hash,
            result_rows,
            source_episode_id,
            trajectory_blob_hash,
            trajectory_encoding,
            trajectory_payload,
            db_blob_hash,
            db_encoding,
            db_payload,
        ]
        if self.has_bindings:
            columns.append("bindings_json")
            values.append(
                canonical_json(bindings or ([self.binding] if self.binding else []))
            )
        placeholders = ",".join("?" for _ in columns)
        before = self.db.total_changes
        self.db.execute(
            f"INSERT OR IGNORE INTO replay_cache({','.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )
        inserted = self.db.total_changes > before
        requested_bindings = bindings or ([self.binding] if self.binding else [])
        if self.has_bindings and requested_bindings:
            current = self.db.execute(
                "SELECT bindings_json FROM replay_cache WHERE cache_id=?",
                (cache_id,),
            ).fetchone()
            existing = json.loads(str(current[0] or "[]"))
            changed = False
            for binding in requested_bindings:
                if binding not in existing:
                    existing.append(binding)
                    changed = True
            if changed:
                self.db.execute(
                    "UPDATE replay_cache SET bindings_json=? WHERE cache_id=?",
                    (canonical_json(existing), cache_id),
                )
        self.db.commit()
        return cache_id, inserted

    def _decode_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        stored_trajectory = decode_payload(
            str(item["trajectory_encoding"]), item["trajectory_payload"]
        )
        item["trajectory"] = canonical_trajectory(stored_trajectory)
        # Preserve the immutable on-disk identity for audits while exposing a
        # canonical identity for current cache matching. Legacy field names
        # intentionally produce a different semantic hash after normalization.
        item["stored_trajectory_hash"] = str(item["trajectory_hash"])
        item["canonical_trajectory_hash"] = semantic_trajectory_hash(item["trajectory"])
        item["db_events"] = decode_payload(
            str(item["db_events_encoding"]), item["db_events_payload"]
        )
        item["query_wall_ms"] = float(item["first_runtime_ms"])
        item["charged_wall_ms"] = float(item["charged_runtime_ms"])
        return item

    def iter_executions(
        self,
        *,
        query_ids: Optional[Iterable[str]] = None,
        cutoff_ms: Optional[int] = None,
        action_config_hash: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        selected_query_ids = sorted({str(value) for value in query_ids or []})
        if selected_query_ids:
            clauses.append(
                "query_id IN (" + ",".join("?" for _ in selected_query_ids) + ")"
            )
            params.extend(selected_query_ids)
        if cutoff_ms is not None:
            clauses.append("created_at_ms<=?")
            params.append(int(cutoff_ms))
        if action_config_hash is not None:
            clauses.append("action_config_hash=?")
            params.append(str(action_config_hash))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.execute(
            f"SELECT * FROM replay_cache {where} " "ORDER BY created_at_ms, cache_id",
            params,
        )
        for row in rows:
            yield self._decode_row(row)

    def trajectory_cache_candidates(self, *, sql_hash: str) -> list[dict[str, Any]]:
        clauses = ["sql_hash=?"]
        params: list[Any] = [str(sql_hash)]
        if self.action_config_hash is not None:
            clauses.append("action_config_hash=?")
            params.append(self.action_config_hash)
        rows = self.db.execute(
            "SELECT * FROM replay_cache WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at_ms DESC, cache_id",
            params,
        ).fetchall()
        eligible = [
            item
            for item in (self._decode_row(row) for row in rows)
            if all(
                decision.get("runtime_source", "physical") != "cache"
                for decision in item["trajectory"]
            )
        ]
        if self.has_bindings and self.binding is not None:
            exact = [
                item
                for item in eligible
                if self.binding in json.loads(str(item.get("bindings_json") or "[]"))
            ]
            if exact:
                eligible = exact
        return eligible

    def expected_result_hash(self, sql_hash: str) -> str | None:
        selected = ["result_hash"]
        if self.has_bindings:
            selected.append("bindings_json")
        rows = self.db.execute(
            f"SELECT {','.join(selected)} FROM replay_cache "
            "WHERE sql_hash=? AND status='ok' AND result_hash IS NOT NULL",
            (str(sql_hash),),
        ).fetchall()
        eligible = rows
        if self.has_bindings and self.binding is not None:
            exact = [
                row
                for row in rows
                if self.binding in json.loads(str(row["bindings_json"] or "[]"))
            ]
            if exact:
                eligible = exact
        hashes = {str(row["result_hash"]) for row in eligible}
        return next(iter(hashes)) if len(hashes) == 1 else None

    def fixed_execution(
        self, *, sql_hash: str, action_config_hash: str
    ) -> dict[str, Any] | None:
        rows = self.db.execute(
            """
            SELECT cache_id, trajectory_hash, status, first_runtime_ms,
                   charged_runtime_ms, timeout_limit_ms, result_hash, result_rows
            FROM replay_cache
            WHERE sql_hash=? AND action_config_hash=?
            ORDER BY created_at_ms, cache_id
            """,
            (str(sql_hash), str(action_config_hash)),
        ).fetchall()
        if not rows:
            return None
        row = dict(rows[0])
        return {
            "cache_id": str(row["cache_id"]),
            "trajectory_hash": str(row["trajectory_hash"]),
            "execution": {
                "status": str(row["status"]),
                "client_wall_ms": float(row["first_runtime_ms"]),
                "charged_wall_ms": float(row["charged_runtime_ms"]),
                "timeout_limit_ms": int(row["timeout_limit_ms"]),
                "result_hash": row["result_hash"],
                "result_rows": row["result_rows"],
            },
        }

    def cache_metadata(self, cache_id: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT cache_id, created_at_ms, timeout_limit_ms, source_episode_id "
            "FROM replay_cache WHERE cache_id=?",
            (str(cache_id),),
        ).fetchone()
        if row is None:
            raise KeyError(cache_id)
        return dict(row)

    def trajectory_cache_summary(self) -> dict[str, Any]:
        row = self.db.execute(
            """
            SELECT COUNT(*) AS entries,
                   COUNT(DISTINCT sql_hash) AS candidate_keys,
                   COALESCE(SUM(first_runtime_ms), 0.0) AS reusable_wall_ms
            FROM replay_cache
            """
        ).fetchone()
        return dict(row)

    def merge_from(self, source_path: str | Path) -> dict[str, int]:
        with ExperienceStore(source_path, read_only=True) as source:
            inserted = 0
            for row in source.iter_executions():
                bindings = (
                    json.loads(str(row.get("bindings_json") or "[]"))
                    if self.has_bindings
                    else None
                )
                _, created = self.append_execution(
                    query_id=str(row["query_id"]),
                    sql_hash=str(row["sql_hash"]),
                    trajectory=row["trajectory"],
                    db_events=row["db_events"],
                    status=str(row["status"]),
                    first_runtime_ms=float(row["first_runtime_ms"]),
                    charged_runtime_ms=float(row["charged_runtime_ms"]),
                    timeout_limit_ms=int(row["timeout_limit_ms"]),
                    action_config_hash=str(row["action_config_hash"]),
                    result_hash=row.get("result_hash"),
                    result_rows=row.get("result_rows"),
                    source_episode_id=str(row["source_episode_id"]),
                    cache_id=str(row["cache_id"]),
                    created_at_ms=int(row["created_at_ms"]),
                    bindings=bindings,
                )
                inserted += int(created)
        return {"replay_cache": inserted}

    def training_summary(
        self, *, query_ids: Optional[Iterable[str]] = None
    ) -> dict[str, Any]:
        entries = list(self.iter_executions(query_ids=query_ids))
        return {
            "episodes": len(entries),
            "timeouts": sum(item["status"] == "timeout" for item in entries),
            "wrong_results": sum(item["status"] == "wrong_result" for item in entries),
            "measured_wall_ms": sum(
                float(item["charged_runtime_ms"]) for item in entries
            ),
            "decisions": sum(len(item["trajectory"]) for item in entries),
            "unique_subquery_action_pairs": len(
                {
                    (decision["state_hash"], canonical_json(decision["action"]))
                    for item in entries
                    for decision in item["trajectory"]
                }
            ),
        }
