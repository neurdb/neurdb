#!/usr/bin/env python3
"""
NeurQO online training bridge.

This process consumes DB-side trajectory JSONL produced by:

    SET neurqo.trajectory_log = '/tmp/neurqo_runtime.jsonl';

Each DB event is converted into a transition:

    state, action, reward, done, next_state=None, timing_ms

The default reward is negative round total time. This is intentionally simple:
it gives us an online data path immediately while keeping model-specific
training outside the DB critical path.

Optionally pass --trainer-module module_or_path:callable. The callable receives
a list[dict] batch and may update a model checkpoint however it wants.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


def load_callable(spec: str | None) -> Callable[[list[dict[str, Any]]], Any] | None:
    if not spec:
        return None
    module_name, sep, attr = spec.partition(":")
    if not sep or not attr:
        raise ValueError("--trainer-module must be 'module_or_path:callable'")

    module_path = Path(module_name)
    if module_path.exists():
        import_name = f"neurqo_online_trainer_{abs(hash(str(module_path)))}"
        mod_spec = importlib.util.spec_from_file_location(import_name, module_path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot import trainer module from {module_path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def event_to_transition(event: dict[str, Any]) -> dict[str, Any] | None:
    phase = event.get("phase")
    if phase not in {"split", "final"}:
        return None
    timing = event.get("timing_ms") or {}
    total_ms = float(timing.get("total") or 0.0)
    return {
        "run_id": event.get("run_id"),
        "round": event.get("round"),
        "state": event.get("state"),
        "action": event.get("action"),
        "reward": -total_ms,
        "done": bool(event.get("stop")) or phase == "final",
        "next_state": None,
        "timing_ms": timing,
        "result": event.get("result"),
        "source_event": phase,
    }


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        lines = f.readlines()
        return lines, f.tell()


def process_lines(
    lines: list[str],
    *,
    out_path: Path,
    trainer: Callable[[list[dict[str, Any]]], Any] | None,
    batch: list[dict[str, Any]],
    batch_size: int,
    seen: set[tuple[Any, Any, str]],
) -> int:
    wrote = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        key = (event.get("run_id"), event.get("round"), event.get("phase"))
        if key in seen:
            continue
        seen.add(key)
        transition = event_to_transition(event)
        if transition is None:
            continue
        append_jsonl(out_path, transition)
        batch.append(transition)
        wrote += 1
        if trainer is not None and len(batch) >= batch_size:
            trainer(list(batch))
            batch.clear()
    return wrote


def main() -> int:
    ap = argparse.ArgumentParser(description="NeurQO online training bridge")
    ap.add_argument("--db-log", required=True, help="DB-side neurqo.trajectory_log JSONL")
    ap.add_argument("--out", required=True, help="transition JSONL output path")
    ap.add_argument("--trainer-module", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    db_log = Path(args.db_log)
    out_path = Path(args.out)
    trainer = load_callable(args.trainer_module)
    offset = 0
    seen: set[tuple[Any, Any, str]] = set()
    batch: list[dict[str, Any]] = []

    while True:
        lines, offset = read_new_lines(db_log, offset)
        wrote = process_lines(
            lines,
            out_path=out_path,
            trainer=trainer,
            batch=batch,
            batch_size=max(args.batch_size, 1),
            seen=seen,
        )
        if args.once:
            if trainer is not None and batch:
                trainer(list(batch))
                batch.clear()
            print(f"processed {wrote} new transitions", flush=True)
            return 0
        if wrote:
            print(f"processed {wrote} new transitions", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
