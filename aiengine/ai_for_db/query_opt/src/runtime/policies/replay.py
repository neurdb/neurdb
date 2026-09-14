"""Replay a previously recorded NQO state-to-Action mapping.

The mapping path is supplied through ``NQO_REPLAY_POLICY_MAP``.  This is
used for fresh physical measurements of an already selected checkpoint: the
saved Actions are reproduced without running the neural network again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

EPHEMERAL_STATE_FIELDS = {
    "pid",
    "run_id",
    "relid",
    "cumulative_cost_ms",
    "plan_state_ms",
    "remaining_splits",
}
TEMP_RELATION_PATTERN = re.compile(r"\btemp[0-9]+\b", re.IGNORECASE)


def _stable_state(value: Any, temp_relations: dict[str, str] | None = None) -> Any:
    if temp_relations is None:
        temp_relations = {}
    if isinstance(value, dict):
        return {
            key: _stable_state(item, temp_relations)
            for key, item in value.items()
            if key not in EPHEMERAL_STATE_FIELDS
        }
    if isinstance(value, list):
        return [_stable_state(item, temp_relations) for item in value]
    if isinstance(value, str):

        def canonical_temp(match: re.Match[str]) -> str:
            key = match.group(0).lower()
            if key not in temp_relations:
                temp_relations[key] = f"__nqo_temp_{len(temp_relations) + 1}"
            return temp_relations[key]

        return TEMP_RELATION_PATTERN.sub(canonical_temp, value)
    return value


def _state_hash(state: dict[str, Any]) -> str:
    # The hierarchical controller normalizes plan_json before invoking a policy.
    # The DB-side semantic state hash was computed from the original compact
    # plan.  Restore that representation when it is available.
    raw_state = dict(state)
    if "db_plan_json" in raw_state:
        raw_state["plan_json"] = raw_state.pop("db_plan_json")
    encoded = json.dumps(
        _stable_state(raw_state),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_mapping() -> dict[str, Any]:
    path = os.environ.get("NQO_REPLAY_POLICY_MAP")
    if not path:
        raise RuntimeError("NQO_REPLAY_POLICY_MAP is not set")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = payload.get("queries")
    owners = payload.get("initial_state_owners")
    if not isinstance(queries, dict) or not isinstance(owners, dict):
        raise ValueError(f"invalid replay policy mapping: {path}")
    return {"queries": queries, "initial_state_owners": owners}


ACTION_MAP = _load_mapping()
PID_TO_QUERY: dict[int, str] = {}


def predict(state: dict[str, Any]) -> dict[str, Any]:
    state_hash = _state_hash(state)
    pid = int(state.get("pid") or 0)
    query_id = PID_TO_QUERY.get(pid)
    if query_id is None:
        owners = ACTION_MAP["initial_state_owners"].get(state_hash) or []
        if not owners:
            raise KeyError(
                f"saved query missing for request_type={state.get('request_type')!r} "
                f"state_hash={state_hash}"
            )
        query_id = str(owners[0])
        PID_TO_QUERY[pid] = query_id
    query = ACTION_MAP["queries"].get(query_id) or {}
    action = (query.get("actions") or {}).get(state_hash)
    if action is None:
        fallback_key = f"{int(state.get('round') or 0)}:{state.get('request_type')}"
        action = (query.get("fallback_actions") or {}).get(fallback_key)
    if action is None:
        request_type = state.get("request_type")
        raise KeyError(
            f"saved Action missing for query_id={query_id!r} "
            f"request_type={request_type!r} "
            f"state_hash={state_hash}"
        )
    result = dict(action)
    result["note"] = "saved checkpoint Action replay"
    result["policy_version"] = "saved-action-replay"
    return result
