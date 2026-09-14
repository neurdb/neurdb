"""Incrementally join policy and DB logs; only this worker writes experience."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from experience.store import ExperienceStore, content_hash
from optimization.actions import ingest_trajectory


def default_data_dir() -> Path:
    if os.environ.get("NQO_DATA_DIR"):
        return Path(os.environ["NQO_DATA_DIR"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout / "data"
    return Path.home() / ".local" / "share" / "neurdb" / "query_opt"


class ExperienceCollector(threading.Thread):
    def __init__(
        self,
        *,
        dataset: str,
        database: str,
        policy_log: Path,
        db_log: Path,
        buffer: Path,
        checkpoint: Path,
        bootstrap: Path | None = None,
        interval: float = 2.0,
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(name="nqo-experience-collector", daemon=True)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("collector interval must be positive and finite")
        self.dataset = dataset
        self.database = database
        self.paths = {"policy": policy_log.resolve(), "db": db_log.resolve()}
        self.buffer = buffer.resolve()
        self.checkpoint = checkpoint.resolve()
        self.bootstrap = bootstrap.resolve() if bootstrap else None
        if len({*self.paths.values(), self.buffer, self.checkpoint}) != 4:
            raise ValueError("logs, buffer, and checkpoint must use different paths")
        if self.bootstrap == self.buffer:
            raise ValueError("the runtime buffer must not overwrite its bootstrap")
        self.interval = interval
        self.log = log
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.startup_error: Exception | None = None
        self.last_error = ""
        self.inserted = 0
        self.state: dict[str, Any] = {}

    def _identity(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset": self.dataset,
            "database": self.database,
            "buffer": str(self.buffer),
            "paths": {key: str(path) for key, path in self.paths.items()},
        }

    def _load_checkpoint(self) -> None:
        if self.checkpoint.exists():
            state = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            if state.get("identity") != self._identity():
                raise ValueError("collector checkpoint belongs to different inputs")
            self.state = state
        else:
            self.state = {"identity": self._identity(), "positions": {}, "pending": {}}

    def _save_checkpoint(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.checkpoint)

    def _read(self, source: str) -> tuple[list[dict[str, Any]], bool]:
        path = self.paths[source]
        if not path.exists():
            return [], True
        events = []
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            identity = [stat.st_dev, stat.st_ino]
            position = self.state["positions"].setdefault(
                source, {"identity": identity, "offset": 0}
            )
            if position["identity"] != identity or stat.st_size < position["offset"]:
                self.log(
                    f"collector: {source} log replaced/truncated; reading from start"
                )
                position.update(identity=identity, offset=0)
            handle.seek(position["offset"])
            for _ in range(500):
                line = handle.readline()
                if not line:
                    return events, True
                if not line.endswith(b"\n"):
                    return events, False
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object event in {path}")
                events.append(value)
                position["offset"] = handle.tell()
        return events, False

    def _accept(self, source: str, event: dict[str, Any]) -> None:
        if source == "policy":
            if event.get("phase") != "policy_decision":
                return
            identity = event.get("state") or {}
        else:
            if event.get("phase") not in {
                "query_start",
                "split",
                "final",
                "query_complete",
            }:
                return
            identity = event
        if identity.get("pid") is None or identity.get("run_id") is None:
            return
        key = f"{identity['pid']}:{identity['run_id']}"
        pending = self.state["pending"].setdefault(key, {"policy": [], "db": []})
        # Log replay/rotation must not duplicate a decision in a pending episode.
        if event not in pending[source]:
            pending[source].append(event)

    def _persist(
        self, store: ExperienceStore, key: str, pending: dict[str, Any]
    ) -> bool:
        start = next((e for e in pending["db"] if e["phase"] == "query_start"), None)
        finish = next(
            (e for e in pending["db"] if e["phase"] == "query_complete"), None
        )
        if start is None or finish is None:
            return False
        if start.get("database") != self.database:
            return True
        if finish.get("database") != self.database:
            raise ValueError(f"database mismatch in execution {key}")
        status = finish["status"]
        if status not in {"ok", "timeout", "error", "cancelled"}:
            raise ValueError(f"invalid execution status: {status}")
        wall_ms = float(finish["wall_ms"])
        if not math.isfinite(wall_ms) or wall_ms < 0:
            raise ValueError("invalid DB runtime")
        rounds = [e for e in pending["db"] if e["phase"] in {"split", "final"}]
        policy_keys = {
            (int(e["state"].get("round", 0)), e["state"].get("request_type"))
            for e in pending["policy"]
        }
        required = {
            (int(e.get("round", 0)), phase)
            for e in rounds
            for phase, state in (e.get("decision_states") or {}).items()
            if isinstance(state, dict)
        }
        if not required.issubset(policy_keys):
            return False
        sql_hash = content_hash(start["sql"].strip())
        known = store.db.execute(
            "SELECT query_id FROM replay_cache WHERE sql_hash=? LIMIT 1", (sql_hash,)
        ).fetchone()
        query_id = str(known[0]) if known else f"sql_{sql_hash[:16]}"
        config_hash = content_hash(
            {
                "dataset": self.dataset,
                "database": self.database,
                "runtime_scope": "db_nqo",
                "action_settings": start.get("action_settings") or {},
            }
        )
        episode_id = f"db_{content_hash([self.database, key])}"
        timeout_ms = int(start["timeout_limit_ms"])
        collected = ingest_trajectory(
            episode_id=episode_id,
            environment_hash=self.dataset,
            implementation_version="db-log-collector-v1",
            db_events=rounds,
            policy_events=pending["policy"],
            timeout_limit_ms=timeout_ms,
            timeout_charged_ms=wall_ms,
            episode_status=status,
        )
        # Raw policy events retain any decisions interrupted before a round ended.
        terminal = dict(finish, dataset=self.dataset, policy_events=pending["policy"])
        _, inserted = store.append_execution(
            query_id=query_id,
            sql_hash=sql_hash,
            trajectory=collected["trajectory"],
            db_events=[e for e in pending["db"] if e["phase"] != "query_complete"]
            + [terminal],
            status=status,
            first_runtime_ms=wall_ms,
            charged_runtime_ms=wall_ms,
            timeout_limit_ms=timeout_ms,
            action_config_hash=config_hash,
            result_rows=finish.get("result_rows"),
            source_episode_id=episode_id,
            cache_id=content_hash([self.dataset, episode_id]),
            created_at_ms=int(finish["ts_ms"]),
        )
        self.inserted += int(inserted)
        return True

    def poll_once(self, store: ExperienceStore) -> None:
        dirty = False
        policy_eof = False
        # A policy line is flushed before PG can finish the corresponding work.
        # Read DB first so its completion cannot overtake our policy read.
        for source in ("db", "policy"):
            events, eof = self._read(source)
            if source == "policy":
                policy_eof = eof
            dirty |= bool(events)
            for event in events:
                self._accept(source, event)
        # Drain policy records before finalizing, including the interrupted round.
        if policy_eof:
            for key, pending in list(self.state["pending"].items()):
                if self._persist(store, key, pending):
                    del self.state["pending"][key]
                    dirty = True
        if dirty:
            self._save_checkpoint()

    def run(self) -> None:
        try:
            self.buffer.parent.mkdir(parents=True, exist_ok=True)
            with self.buffer.with_suffix(".collector.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self.checkpoint.exists() and not self.buffer.exists():
                    raise ValueError(
                        "buffer is missing but a collector checkpoint exists"
                    )
                if (
                    not self.buffer.exists()
                    and self.bootstrap
                    and self.bootstrap.is_file()
                ):
                    temporary = self.buffer.with_suffix(".bootstrap.tmp")
                    shutil.copyfile(self.bootstrap, temporary)
                    os.replace(temporary, self.buffer)
                with ExperienceStore(self.buffer) as store:
                    store.db.execute("PRAGMA busy_timeout=1000")
                    self._load_checkpoint()
                    self.ready.set()
                    while True:
                        try:
                            self.poll_once(store)
                            self.last_error = ""
                        except Exception as exc:
                            self.last_error = str(exc)
                            self.log(f"collector error (will retry): {exc}")
                            store.db.rollback()
                            self._load_checkpoint()
                        if self.stopping.wait(self.interval):
                            self.poll_once(store)
                            break
        except Exception as exc:
            self.last_error = str(exc)
            if not self.ready.is_set():
                self.startup_error = exc
            self.log(f"collector stopped: {exc}")
        finally:
            self.ready.set()

    def stop(self) -> None:
        self.stopping.set()
        self.join(timeout=5)
        if self.is_alive():
            self.log(
                "collector shutdown timed out; uncheckpointed logs will be replayed"
            )
