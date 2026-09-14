#!/usr/bin/env python3
"""Shared runtime utilities for online NQO collection and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from experience.store import canonical_json, content_hash
from optimization.action_vocabulary import (
    ADAPT_PHASE,
    DEC_PHASE,
    DECISION_PHASES,
    ENUM_PHASE,
    SCHED_PHASE,
    canonical_ajoin_action,
    canonical_dec_action,
    canonical_enum_action,
    canonical_filter_action,
    canonical_phase,
    normalize_policy_action,
    normalize_policy_state,
)

ONLINE_ACTION_SPACE_VERSION = "dec2-sched3-enum2-adapt4-v3"
STATE_CONTRACT_FORBIDDEN_FIELDS = {
    DEC_PHASE: {
        "plan_available",
        "plan_json",
        "plan_rows",
        "plan_summary",
        "plan_total_cost",
        "plan_width",
    },
    ENUM_PHASE: {
        "plan_available",
        "plan_json",
        "plan_rows",
        "plan_summary",
        "plan_total_cost",
        "plan_width",
    },
    ADAPT_PHASE: {
        "aliases",
        "candidates",
        "original_sql",
        "relations",
        "sql",
    },
}
ACTION_CONFIG_SCHEMA_VERSION = 1
STATEMENT_TIMEOUT_MS = 60_000
TIMEOUT_CHARGE_FACTOR = 5.0
TIMEOUT_CHARGE_CAP_MS = 360_000
DATASET_TIMEOUT_CAP_MS = {
    "JOB": 60_000,
    "STACK": 60_000,
    "TPCH": 60_000,
}
REQUIRED_ACTION_PARAMETERS = {
    "sched_alpha",
    "max_rounds",
    "search_max_rels",
    "search_exact_cardinality",
    "aja_conservative_rows",
    "aja_aggressive_rows",
    "aja_max_nestloop_cost_ratio_pct",
    "aja_aggressive_max_nestloop_cost_ratio_pct",
    "lip_max_build_relation_rows",
    "lip_selective_plan_rows",
    "lip_max_build_selectivity_pct",
    "lip_min_probe_ratio",
    "lip_max_filters",
}
EPHEMERAL_STATE_FIELDS = {
    "pid",
    "run_id",
    "relid",
    "cumulative_cost_ms",
    "plan_state_ms",
    # The executor sets this to zero when a fixed prefix-depth cap is reached.
    # The residual SQL/graph still identifies the semantic state.
    "remaining_splits",
}
# Request-local identifiers do not affect policy inference. Unlike
# ``EPHEMERAL_STATE_FIELDS``, this set deliberately retains dynamic context
# consumed by the model or its action masks (notably cumulative_cost_ms and
# remaining_splits). It is used only to index immutable cached states; it does
# not change historical semantic hashes stored in experience SQLite.
MODEL_INPUT_EPHEMERAL_STATE_FIELDS = {
    "pid",
    "run_id",
    "relid",
    "plan_state_ms",
}
TEMP_RELATION_PATTERN = re.compile(r"\btemp[0-9]+\b", re.IGNORECASE)

LEGACY_PROFILE_FIELDS = {
    "high": "dec",
    "split_rounds": "dec_rounds",
    "schedule_alpha": "sched_alpha",
    "schedule_alpha_sequence": "sched_alpha_sequence",
    "search": "enum",
    "search_k": "enum_k",
    "lip": "filter",
    "aja": "ajoin",
}


@dataclass(frozen=True)
class ActionProfile:
    """A fixed policy used to isolate one mechanism during calibration."""

    name: str
    nqo_enabled: bool = True
    dec: str = "skip"
    dec_rounds: int = -1
    sched_alpha: float = 0.5
    sched_alpha_sequence: tuple[float, ...] = ()
    enum: str = "native"
    enum_k: int = 5
    filter: str = "none"
    ajoin: str = "off"
    max_rounds: int = 16
    search_max_rels: int = 12
    search_exact_cardinality: bool = False
    aja_conservative_rows: int = 362_443
    aja_aggressive_rows: int = 3_624_434
    aja_max_nestloop_cost_ratio_pct: int = 150
    aja_aggressive_max_nestloop_cost_ratio_pct: int = 125
    lip_max_build_relation_rows: int = 500_000
    lip_selective_plan_rows: int = 10_000
    lip_max_build_selectivity_pct: int = 10
    lip_min_probe_ratio: int = 2
    lip_max_filters: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "dec", canonical_dec_action(self.dec))
        object.__setattr__(self, "enum", canonical_enum_action(self.enum, self.enum_k))
        object.__setattr__(self, "filter", canonical_filter_action(self.filter))
        object.__setattr__(self, "ajoin", canonical_ajoin_action(self.ajoin))
        if not 0.0 <= self.sched_alpha <= 1.0:
            raise ValueError("sched_alpha must be in [0, 1]")
        if any(alpha < 0.0 or alpha > 1.0 for alpha in self.sched_alpha_sequence):
            raise ValueError("sched_alpha_sequence values must be in [0, 1]")

    @property
    def is_postgres(self) -> bool:
        return not self.nqo_enabled

    def policy_environment(self) -> dict[str, str]:
        environment = {
            "NQO_FIXED_DEC": self.dec,
            "NQO_FIXED_DEC_ROUNDS": str(self.dec_rounds),
            "NQO_FIXED_SCHED_ALPHA": str(self.sched_alpha),
            "NQO_FIXED_ENUM": self.enum,
            "NQO_FIXED_ENUM_K": str(self.enum_k),
            "NQO_FIXED_FILTER": self.filter,
            "NQO_FIXED_AJOIN": self.ajoin,
        }
        if self.sched_alpha_sequence:
            environment["NQO_FIXED_SCHED_ALPHA_SEQUENCE"] = ",".join(
                str(alpha) for alpha in self.sched_alpha_sequence
            )
        return environment

    def guc_settings(self) -> dict[str, Any]:
        return {
            "nqo.max_rounds": self.max_rounds,
            "nqo.search_topk": self.enum_k,
            "nqo.search_max_rels": self.search_max_rels,
            "nqo.search_exact_cardinality": (self.search_exact_cardinality),
            "nqo.aja_conservative_rows": self.aja_conservative_rows,
            "nqo.aja_aggressive_rows": self.aja_aggressive_rows,
            "nqo.aja_max_nestloop_cost_ratio_pct": (
                self.aja_max_nestloop_cost_ratio_pct
            ),
            "nqo.aja_aggressive_max_nestloop_cost_ratio_pct": (
                self.aja_aggressive_max_nestloop_cost_ratio_pct
            ),
            "nqo.lip_max_build_relation_rows": (self.lip_max_build_relation_rows),
            "nqo.lip_selective_plan_rows": (self.lip_selective_plan_rows),
            "nqo.lip_max_build_selectivity_pct": (self.lip_max_build_selectivity_pct),
            "nqo.lip_min_probe_ratio": self.lip_min_probe_ratio,
            "nqo.lip_max_filters": self.lip_max_filters,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def legacy_dict(self) -> dict[str, Any]:
        """Return the v2 profile identity used by released buffers."""
        legacy = self.to_dict()
        legacy.update(
            {
                "high": "split" if self.dec == "apply" else "stop",
                "split_rounds": self.dec_rounds,
                "schedule_alpha": self.sched_alpha,
                "schedule_alpha_sequence": self.sched_alpha_sequence,
                "search": (
                    "default"
                    if self.enum == "native"
                    else "split" if self.enum == "top1" else self.enum
                ),
                "search_k": self.enum_k,
                "lip": self.filter,
                "aja": "none" if self.ajoin == "off" else self.ajoin,
            }
        )
        for canonical in LEGACY_PROFILE_FIELDS.values():
            legacy.pop(canonical, None)
        return legacy

    def compatible_hashes(self) -> tuple[str, ...]:
        """Return canonical and released-v2 profile hashes, in preference order."""
        hashes = (content_hash(self.to_dict()), content_hash(self.legacy_dict()))
        return tuple(dict.fromkeys(hashes))

    @classmethod
    def from_mapping(cls, *, name: str, values: dict[str, Any]) -> "ActionProfile":
        """Load a profile while accepting released legacy field names."""
        normalized = {
            LEGACY_PROFILE_FIELDS.get(key, key): value for key, value in values.items()
        }
        return cls(name=name, **normalized)


def builtin_profiles() -> dict[str, ActionProfile]:
    """Return independent-action profiles used by the calibration protocol."""
    return {
        "pg": ActionProfile(name="pg", nqo_enabled=False),
        "nqo_none": ActionProfile(name="nqo_none"),
        "query_split": ActionProfile(name="query_split", dec="split"),
        "split_search": ActionProfile(name="split_search", enum="split"),
        "top5": ActionProfile(name="top5", enum="top5", enum_k=5),
        "top5_lip_selective": ActionProfile(
            name="top5_lip_selective",
            enum="top5",
            enum_k=5,
            filter="selective",
        ),
        "top5_aja_conservative": ActionProfile(
            name="top5_aja_conservative",
            enum="top5",
            enum_k=5,
            ajoin="conservative",
        ),
        "top5_lip_selective_aja_conservative": ActionProfile(
            name="top5_lip_selective_aja_conservative",
            enum="top5",
            enum_k=5,
            filter="selective",
            ajoin="conservative",
        ),
        "top10": ActionProfile(name="top10", enum="top10", enum_k=10),
        "lip_full": ActionProfile(name="lip_full", filter="full"),
        "lip_selective": ActionProfile(name="lip_selective", filter="selective"),
        "aja_conservative": ActionProfile(
            name="aja_conservative", ajoin="conservative"
        ),
        "aja_aggressive": ActionProfile(name="aja_aggressive", ajoin="aggressive"),
        "lip_full_aja_conservative": ActionProfile(
            name="lip_full_aja_conservative",
            filter="full",
            ajoin="conservative",
        ),
        "lip_full_aja_aggressive": ActionProfile(
            name="lip_full_aja_aggressive",
            filter="full",
            ajoin="aggressive",
        ),
        "lip_selective_aja_conservative": ActionProfile(
            name="lip_selective_aja_conservative",
            filter="selective",
            ajoin="conservative",
        ),
        "lip_selective_aja_aggressive": ActionProfile(
            name="lip_selective_aja_aggressive",
            filter="selective",
            ajoin="aggressive",
        ),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def validate_policy_state_contract(
    policy_events: Iterable[dict[str, Any]],
) -> None:
    """Reject cross-layer state leakage before storing experiment data."""
    for event_index, event in enumerate(policy_events):
        if event.get("phase") != "policy_decision":
            continue
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"policy event {event_index} has no state object")
        state = normalize_policy_state(state)
        phase = canonical_phase(state.get("request_type"))
        forbidden = STATE_CONTRACT_FORBIDDEN_FIELDS.get(phase)
        if forbidden is None:
            continue
        leaked = sorted(forbidden & set(state))
        if leaked:
            raise ValueError(f"{phase} state violates hierarchical contract: {leaked}")
        if phase == ADAPT_PHASE:
            if not state.get("plan_available"):
                raise ValueError("Adapt state has no selected physical plan")
            if not isinstance(state.get("plan_json"), dict):
                raise ValueError("Adapt state has no plan_json object")


def hash_result_rows(rows: Iterable[Iterable[Any]]) -> tuple[str, int]:
    """Hash a typed result multiset without retaining all rows in memory.

    SQL does not define row order without ORDER BY.  Combining per-row
    digests commutatively avoids rejecting equivalent plans just because they
    emit rows in a different physical order.  The sum and sum-of-squares keep
    duplicate multiplicity; SHA-256 makes an accidental collision negligible.
    """
    modulus = 1 << 256
    digest_sum = 0
    digest_sum_squares = 0
    digest_xor = 0
    count = 0
    for row in rows:
        encoded = canonical_json(
            [
                {
                    "type": type(value).__name__,
                    "value": value,
                }
                for value in row
            ]
        ).encode("utf-8")
        row_digest = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
        digest_sum = (digest_sum + row_digest) % modulus
        digest_sum_squares = (digest_sum_squares + row_digest * row_digest) % modulus
        digest_xor ^= row_digest
        count += 1
    aggregate = hashlib.sha256()
    aggregate.update(count.to_bytes(8, "big"))
    aggregate.update(digest_sum.to_bytes(32, "big"))
    aggregate.update(digest_sum_squares.to_bytes(32, "big"))
    aggregate.update(digest_xor.to_bytes(32, "big"))
    return aggregate.hexdigest(), count


def dynamic_timeout_ms(
    pg_baseline_ms: float,
    *,
    factor: float = 2.0,
    slack_ms: int = 2_000,
    minimum_ms: int = 5_000,
    maximum_ms: int = 60_000,
) -> int:
    return max(
        minimum_ms,
        min(maximum_ms, int(math.ceil(factor * pg_baseline_ms + slack_ms))),
    )


def timeout_charged_runtime_ms(
    pg_baseline_ms: float,
    *,
    factor: float = TIMEOUT_CHARGE_FACTOR,
    cap_ms: int = TIMEOUT_CHARGE_CAP_MS,
) -> float:
    """Return the metric/reward charge for a query stopped at 60 seconds."""
    if pg_baseline_ms <= 0.0:
        raise ValueError("PG baseline runtime must be positive")
    if factor <= 0.0 or cap_ms <= 0:
        raise ValueError("timeout charge factor and cap must be positive")
    return min(float(cap_ms), factor * float(pg_baseline_ms))


def first_runtime_timeout_ms(
    pg_first_ms: float,
    *,
    factor: float = TIMEOUT_CHARGE_FACTOR,
    cap_ms: int = 60_000,
) -> int:
    """Return ceil(min(factor * PG-first, cap)) for statement_timeout."""
    return max(
        1,
        int(
            math.ceil(
                timeout_charged_runtime_ms(
                    pg_first_ms,
                    factor=factor,
                    cap_ms=cap_ms,
                )
            )
        ),
    )


def dataset_timeout_cap_ms(workload: str) -> int:
    try:
        return DATASET_TIMEOUT_CAP_MS[workload.strip().upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported workload {workload!r}") from exc


def global_experience_path(pgdb_root: Path, workload: str) -> Path:
    """Return the shared lightweight execution buffer for a dataset."""
    dataset = workload.strip().upper()
    if dataset not in DATASET_TIMEOUT_CAP_MS:
        raise ValueError(f"unsupported workload {workload!r}")
    return (
        Path(pgdb_root).resolve()
        / ".nqo_runtime"
        / "experience"
        / f"{dataset.lower()}_light.sql"
    )


def action_config_hash(config: dict[str, Any]) -> str:
    """Hash the immutable fields of a dataset-level Action configuration."""
    payload = {
        "schema_version": config.get("schema_version"),
        "workload": str(config.get("workload") or "").upper(),
        "parameters": config.get("parameters"),
        "execution_protocol": config.get("execution_protocol"),
    }
    if "profile_parameters" in config:
        payload["profile_parameters"] = config.get("profile_parameters")
    if "selected_profiles" in config:
        payload["selected_profiles"] = config.get("selected_profiles")
    if "excluded_families" in config:
        payload["excluded_families"] = config.get("excluded_families")
    return content_hash(payload)


def load_action_config(
    path: Path,
    *,
    workload: Optional[str] = None,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Action config must be a JSON object: {path}")
    if config.get("schema_version") != ACTION_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Action config schema in {path}: "
            f"{config.get('schema_version')!r}"
        )
    if config.get("status") != "frozen":
        raise ValueError(f"Action config is not frozen: {path}")
    config_workload = str(config.get("workload") or "").upper()
    if workload is not None and config_workload != workload.strip().upper():
        raise ValueError(
            f"Action config workload {config_workload!r} does not match "
            f"{workload.strip().upper()!r}"
        )
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Action config has no parameter object: {path}")
    expected_hash = action_config_hash(config)
    if config.get("config_hash") != expected_hash:
        raise ValueError(
            f"Action config hash mismatch in {path}: expected {expected_hash}"
        )
    # Validate the immutable file before translating legacy field names.  The
    # returned object is canonical, while its original config_hash remains the
    # identity of the read-only released artifact.
    normalized_parameters = {
        LEGACY_PROFILE_FIELDS.get(key, key): value for key, value in parameters.items()
    }
    missing = sorted(REQUIRED_ACTION_PARAMETERS - set(normalized_parameters))
    if missing:
        raise ValueError(f"Action config is missing parameters: {missing}")
    normalized = dict(config)
    normalized["parameters"] = normalized_parameters
    return normalized


def apply_action_config(
    namespace: Any,
    config: dict[str, Any],
) -> None:
    """Apply frozen parameters to a benchmark/training argument namespace."""
    parameters = config["parameters"]
    effective_parameters = dict(parameters)
    requested_profiles = [
        item.strip()
        for item in str(getattr(namespace, "profiles", "")).split(",")
        if item.strip()
    ]
    profile_parameters = config.get("profile_parameters") or {}
    if len(requested_profiles) == 1:
        effective_parameters.update(
            {
                key: value
                for key, value in profile_parameters.get(
                    requested_profiles[0], {}
                ).items()
                if value is not None
            }
        )
    for key, value in effective_parameters.items():
        target = LEGACY_PROFILE_FIELDS.get(key, key)
        if (
            target == "sched_alpha"
            and not hasattr(namespace, target)
            and hasattr(namespace, "initial_sched_alpha")
        ):
            target = "initial_sched_alpha"
        if hasattr(namespace, target):
            setattr(namespace, target, value)
    namespace.action_config_hash = config["config_hash"]


def _identity(state: dict[str, Any], phase: str) -> tuple[int, int, int, str]:
    return (
        int(state.get("pid") or 0),
        int(state.get("run_id") or 0),
        int(state.get("round") or 0),
        phase,
    )


def _policy_decision_index(
    policy_events: Iterable[dict[str, Any]],
) -> dict[tuple[int, int, int, str], list[dict[str, Any]]]:
    index: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    for event in policy_events:
        if event.get("phase") != "policy_decision":
            continue
        state = event.get("state") or {}
        state = normalize_policy_state(state)
        phase = canonical_phase(state.get("request_type"))
        if phase not in DECISION_PHASES:
            continue
        index.setdefault(_identity(state, phase), []).append(event)
    return index


def _fallback_phase_action(phase: str, combined: dict[str, Any]) -> dict[str, Any]:
    combined = normalize_policy_action(combined, phase=phase)
    if phase == DEC_PHASE:
        return {
            "dec_action": combined.get("dec_action"),
            "order_decision": combined.get("order_decision"),
        }
    if phase == SCHED_PHASE:
        return {
            "candidate_id": combined.get("candidate_id"),
            "sched_alpha": combined.get("sched_alpha"),
            "selection_strategy": combined.get("selection_strategy"),
        }
    if phase == ENUM_PHASE:
        return {
            "enum_action": combined.get("enum_action"),
            "enum_k": combined.get("enum_k"),
        }
    return {
        "ajoin_action": combined.get("ajoin_action"),
        "filter_action": combined.get("filter_action"),
    }


def stable_state(
    value: Any,
    temp_relations: Optional[dict[str, str]] = None,
) -> Any:
    if temp_relations is None:
        temp_relations = {}
    if isinstance(value, dict):
        return {
            key: stable_state(item, temp_relations)
            for key, item in value.items()
            if key not in EPHEMERAL_STATE_FIELDS
        }
    if isinstance(value, list):
        return [stable_state(item, temp_relations) for item in value]
    if isinstance(value, str):

        def canonical_temp(match: re.Match[str]) -> str:
            key = match.group(0).lower()
            if key not in temp_relations:
                temp_relations[key] = f"__nqo_temp_{len(temp_relations) + 1}"
            return temp_relations[key]

        return TEMP_RELATION_PATTERN.sub(canonical_temp, value)
    return value


def model_input_state(
    value: Any,
    temp_relations: Optional[dict[str, str]] = None,
) -> Any:
    """Canonicalize every request field that can affect policy inference.

    ``stable_state`` intentionally ignores execution context so semantically
    equivalent states share training labels. Cache replay has a stricter
    requirement: requests must not collapse when model context or action masks
    differ. This form removes only request-local identifiers while retaining
    cumulative time, split budget, round information, SQL/plan payloads, and
    candidate information.
    """
    if temp_relations is None:
        temp_relations = {}
    if isinstance(value, dict):
        return {
            key: model_input_state(item, temp_relations)
            for key, item in value.items()
            if key not in MODEL_INPUT_EPHEMERAL_STATE_FIELDS
        }
    if isinstance(value, list):
        return [model_input_state(item, temp_relations) for item in value]
    if isinstance(value, str):

        def canonical_temp(match: re.Match[str]) -> str:
            key = match.group(0).lower()
            if key not in temp_relations:
                temp_relations[key] = f"__nqo_temp_{len(temp_relations) + 1}"
            return temp_relations[key]

        return TEMP_RELATION_PATTERN.sub(canonical_temp, value)
    return value


def model_input_hash(state: dict[str, Any]) -> str:
    """Hash the complete canonical request seen by one policy decision."""
    return content_hash(model_input_state(normalize_policy_state(state)))


def _semantic_action(phase: str, action: dict[str, Any]) -> dict[str, Any]:
    phase = canonical_phase(phase)
    action = normalize_policy_action(action, phase=phase)
    fields = {
        DEC_PHASE: ("dec_action", "order_decision"),
        SCHED_PHASE: (
            "candidate_id",
            "sched_alpha",
        ),
        ENUM_PHASE: ("enum_action", "enum_k"),
        ADAPT_PHASE: ("ajoin_action", "filter_action"),
    }[phase]
    return {key: action.get(key) for key in fields if action.get(key) is not None}


def semantic_policy_trajectory(
    policy_events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return ordered policy states and implementation-neutral actions."""
    trajectory = []
    for event in policy_events:
        if event.get("phase") != "policy_decision":
            continue
        state = event.get("state")
        action = event.get("action")
        if not isinstance(state, dict) or not isinstance(action, dict):
            continue
        state = normalize_policy_state(state)
        phase = canonical_phase(state.get("request_type"))
        if phase not in DECISION_PHASES:
            continue
        trajectory.append(
            {
                "phase": phase,
                "state": state,
                "state_hash": content_hash(stable_state(state)),
                "action": _semantic_action(phase, action),
            }
        )
    return trajectory


def _phase_runtime_ms(phase: str, timing: dict[str, Any]) -> float:
    """Charge each decision the complete round it helped produce.

    Enum and Adapt jointly determine the final executable plan.  Charging only
    planner or executor time would omit action-specific costs such as LIP build,
    adaptive-join probing, replanning, ANALYZE, and residual rewriting.  The
    component timings remain available in the DB-event payload for attribution.
    """
    total = float(timing.get("total") or 0.0)
    planning = float(timing.get("planning") or 0.0)
    execution = float(timing.get("execution") or 0.0)
    return total or planning + execution


def _downstream_signature(phase: str, phase_actions: dict[str, dict[str, Any]]) -> str:
    phase_index = DECISION_PHASES.index(phase)
    downstream = {
        name: _semantic_action(name, phase_actions[name])
        for name in DECISION_PHASES[phase_index + 1 :]
        if name in phase_actions
    }
    return content_hash(downstream) if downstream else ""


def ingest_trajectory(
    *,
    episode_id: str,
    environment_hash: str,
    implementation_version: str,
    db_events: Iterable[dict[str, Any]],
    policy_events: Iterable[dict[str, Any]],
    timeout_limit_ms: int,
    timeout_charged_ms: Optional[float] = None,
    episode_status: str,
    runtime_source: str = "physical",
) -> dict[str, Any]:
    """Build one self-contained trajectory for the single-table buffer.

    Persistence is performed once by the caller after the complete execution
    result is available.
    """
    policy_index = _policy_decision_index(policy_events)
    failure_charge_ms = float(
        timeout_limit_ms if timeout_charged_ms is None else timeout_charged_ms
    )
    round_count = 0
    decision_count = 0
    stored_rounds: set[int] = set()
    trajectory: list[dict[str, Any]] = []

    for event in db_events:
        round_index = int(event.get("round") or 0)
        timing = event.get("timing_ms") or {}
        stored_states = event.get("decision_states") or {}
        states = {
            canonical_phase(phase): normalize_policy_state(state)
            for phase, state in stored_states.items()
            if isinstance(state, dict)
        }
        combined_action = normalize_policy_action(event.get("action") or {})

        round_count += 1
        stored_rounds.add(round_index)

        phase_states = {
            phase: states.get(phase)
            for phase in DECISION_PHASES
            if isinstance(states.get(phase), dict)
        }
        phase_actions: dict[str, dict[str, Any]] = {}
        for phase, state in phase_states.items():
            key = _identity(state, phase)
            matches = policy_index.get(key) or []
            policy_event = matches.pop(0) if matches else None
            if policy_event is not None:
                action = normalize_policy_action(
                    policy_event.get("action") or {}, phase=phase
                )
            else:
                action = _fallback_phase_action(phase, combined_action)
            phase_actions[phase] = action

        for phase, state in phase_states.items():
            action = phase_actions[phase]
            runtime_ms = _phase_runtime_ms(phase, timing)
            is_timeout = episode_status == "timeout"
            trajectory.append(
                {
                    "phase": phase,
                    "state": state,
                    "state_hash": content_hash(stable_state(state)),
                    "action": _semantic_action(phase, action),
                    "round_index": round_index,
                    "runtime_ms": None if is_timeout else runtime_ms,
                    "charged_runtime_ms": (
                        failure_charge_ms if is_timeout else runtime_ms
                    ),
                    "runtime_source": runtime_source,
                    "timeout_limit_ms": timeout_limit_ms,
                    "is_timeout": is_timeout,
                    "observed_at_ms": int(
                        event.get("ts_ms") or time.time_ns() // 1_000_000
                    ),
                    "downstream_signature": _downstream_signature(phase, phase_actions),
                    "environment_hash": environment_hash,
                    "implementation_version": implementation_version,
                    "policy": action,
                }
            )
            decision_count += 1

    if episode_status == "timeout":
        pending_by_round: dict[int, list[dict[str, Any]]] = {}
        for pending in policy_index.values():
            for event in pending:
                state = event.get("state") or {}
                pending_by_round.setdefault(int(state.get("round") or 0), []).append(
                    event
                )

        for round_index, pending in sorted(pending_by_round.items()):
            if round_index not in stored_rounds:
                round_count += 1

            actions = {
                canonical_phase(
                    (event.get("state") or {}).get("request_type")
                ): normalize_policy_action(
                    event.get("action") or {},
                    phase=(event.get("state") or {}).get("request_type"),
                )
                for event in pending
            }
            for event in pending:
                state = normalize_policy_state(event.get("state") or {})
                phase = canonical_phase(state.get("request_type"))
                if phase not in DECISION_PHASES:
                    continue
                action = dict(event.get("action") or {})
                trajectory.append(
                    {
                        "phase": phase,
                        "state": state,
                        "state_hash": content_hash(stable_state(state)),
                        "action": _semantic_action(phase, action),
                        "round_index": round_index,
                        "runtime_ms": None,
                        "charged_runtime_ms": failure_charge_ms,
                        "runtime_source": runtime_source,
                        "timeout_limit_ms": timeout_limit_ms,
                        "is_timeout": True,
                        "observed_at_ms": time.time_ns() // 1_000_000,
                        "downstream_signature": _downstream_signature(phase, actions),
                        "environment_hash": environment_hash,
                        "implementation_version": implementation_version,
                        "policy": action,
                    }
                )
                decision_count += 1

    return {
        "rounds": round_count,
        "decisions": decision_count,
        "trajectory": trajectory,
    }


def query_runtime_summary(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("is_warmup"):
            continue
        grouped.setdefault(str(record["query_id"]), []).append(record)

    summary = {}
    for query_id, samples in grouped.items():
        official = max(
            samples,
            key=lambda item: int(item.get("repetition") or 0),
        )
        official_charged_ms = float(official["charged_wall_ms"])
        materialized_rows = int(official.get("materialized_rows") or 0)
        materialized_bytes = int(official.get("materialized_bytes") or 0)
        split_applied = materialized_rows > 0 or materialized_bytes > 0
        summary[query_id] = {
            "query_id": query_id,
            "samples": len(samples),
            # Keep the legacy key while consumers migrate. Under the online
            # protocol this is the last execution, not a statistical median.
            "median_charged_ms": official_charged_ms,
            "official_charged_ms": official_charged_ms,
            "official_client_wall_ms": float(official.get("client_wall_ms") or 0.0),
            "official_repetition": int(official.get("repetition") or 0),
            "status": str(official.get("status") or "error"),
            "correctness_validation": official.get("correctness_validation"),
            "result_hash": official.get("result_hash"),
            "result_rows": official.get("result_rows"),
            "materialized_rows": materialized_rows,
            "materialized_bytes": materialized_bytes,
            "split_applied": split_applied,
            "search_applied": bool(official.get("search_applied", False)),
            "lip_filters": int(official.get("lip_filters") or 0),
            "aja_decided": int(official.get("aja_decided") or 0),
            "action_applied": bool(
                split_applied
                or official.get("search_applied", False)
                or int(official.get("lip_filters") or 0) > 0
                or int(official.get("aja_decided") or 0) > 0
            ),
            "timeouts": sum(item.get("status") == "timeout" for item in samples),
            "wrong_results": sum(
                item.get("status") == "wrong_result" for item in samples
            ),
            "errors": sum(item.get("status") == "error" for item in samples),
        }
    return summary


def workload_speedup(
    profile: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    expected_query_ids: Optional[Iterable[str]] = None,
) -> Optional[float]:
    expected = (
        list(expected_query_ids)
        if expected_query_ids is not None
        else sorted(set(profile) & set(baseline))
    )
    missing_profile = sorted(set(expected) - set(profile))
    missing_baseline = sorted(set(expected) - set(baseline))
    if missing_profile or missing_baseline:
        raise ValueError(
            "incomplete workload coverage: "
            f"profile={missing_profile}, baseline={missing_baseline}"
        )
    common = expected
    if not common:
        return None
    denominator = sum(profile[qid]["median_charged_ms"] for qid in common)
    if denominator <= 0.0:
        return None
    return sum(baseline[qid]["median_charged_ms"] for qid in common) / denominator


def workload_metrics(
    profile: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    expected_query_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Summarize one official measurement per expected query."""
    common = (
        list(expected_query_ids)
        if expected_query_ids is not None
        else sorted(set(profile) & set(baseline))
    )
    missing_profile = sorted(set(common) - set(profile))
    missing_baseline = sorted(set(common) - set(baseline))
    if missing_profile or missing_baseline:
        raise ValueError(
            "incomplete workload coverage: "
            f"profile={missing_profile}, baseline={missing_baseline}"
        )
    if not common:
        return {
            "query_count": 0,
            "coverage_complete": True,
            "valid": True,
            "pg_total_ms": 0.0,
            "action_total_ms": 0.0,
            "workload_speedup": None,
            "geometric_mean_speedup": None,
            "improved_queries": 0,
            "improved_pct": None,
            "regressed_queries": 0,
            "tied_queries": 0,
            "timeouts": 0,
            "wrong_results": 0,
            "errors": 0,
            "split_applied_queries": 0,
            "split_application_pct": None,
            "search_applied_queries": 0,
            "search_application_pct": None,
            "lip_applied_queries": 0,
            "lip_application_pct": None,
            "aja_applied_queries": 0,
            "aja_application_pct": None,
            "action_applied_queries": 0,
            "action_application_pct": None,
            "per_query_speedups": {},
        }

    pg_total_ms = sum(float(baseline[qid]["median_charged_ms"]) for qid in common)
    action_total_ms = sum(float(profile[qid]["median_charged_ms"]) for qid in common)
    per_query_speedups = {
        qid: (
            float(baseline[qid]["median_charged_ms"])
            / float(profile[qid]["median_charged_ms"])
        )
        for qid in common
        if float(profile[qid]["median_charged_ms"]) > 0.0
    }
    speedups = list(per_query_speedups.values())
    improved = sum(speedup > 1.0 for speedup in speedups)
    regressed = sum(speedup < 1.0 for speedup in speedups)
    tied = len(speedups) - improved - regressed
    timeouts = sum(int(profile[qid].get("timeouts") or 0) for qid in common)
    wrong_results = sum(int(profile[qid].get("wrong_results") or 0) for qid in common)
    errors = sum(int(profile[qid].get("errors") or 0) for qid in common)
    search_applied = sum(bool(profile[qid].get("search_applied")) for qid in common)
    split_applied = sum(bool(profile[qid].get("split_applied")) for qid in common)
    lip_applied = sum(int(profile[qid].get("lip_filters") or 0) > 0 for qid in common)
    aja_applied = sum(int(profile[qid].get("aja_decided") or 0) > 0 for qid in common)
    action_applied = sum(bool(profile[qid].get("action_applied")) for qid in common)
    return {
        "query_count": len(common),
        "coverage_complete": True,
        "valid": wrong_results == 0 and errors == 0,
        "pg_total_ms": pg_total_ms,
        "action_total_ms": action_total_ms,
        "workload_speedup": (
            pg_total_ms / action_total_ms if action_total_ms > 0.0 else None
        ),
        "geometric_mean_speedup": (
            math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups))
            if speedups and all(speedup > 0.0 for speedup in speedups)
            else None
        ),
        "improved_queries": improved,
        "improved_pct": 100.0 * improved / len(common),
        "regressed_queries": regressed,
        "tied_queries": tied,
        "timeouts": timeouts,
        "wrong_results": wrong_results,
        "errors": errors,
        "search_applied_queries": search_applied,
        "search_application_pct": 100.0 * search_applied / len(common),
        "split_applied_queries": split_applied,
        "split_application_pct": 100.0 * split_applied / len(common),
        "lip_applied_queries": lip_applied,
        "lip_application_pct": 100.0 * lip_applied / len(common),
        "aja_applied_queries": aja_applied,
        "aja_application_pct": 100.0 * aja_applied / len(common),
        "action_applied_queries": action_applied,
        "action_application_pct": 100.0 * action_applied / len(common),
        "per_query_speedups": per_query_speedups,
    }
