#!/usr/bin/env python3
"""Train NQO policies from recorded PostgreSQL execution experience."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from experience.store import ExperienceStore, content_hash
from model.encoders.query_graph import CatalogInfo
from model.encoders.state import StructuredState
from model.policy.action_space import (
    ACTION_ABLATIONS,
    ADAPT_LABELS,
    ENUM_LABELS,
    N_ADAPT,
    N_DEC,
    N_ENUM,
    N_SCHED,
    SCHED_ALPHA_VALUES,
    Transition,
    _base_query_family,
    apply_action_ablation_mask,
    migrate_action_space_checkpoint_tensors,
)
from model.policy.hierarchical_actor_critic import HACNetwork, ppo_update
from optimization.action_vocabulary import (
    ADAPT_PHASE,
    DEC_PHASE,
    ENUM_PHASE,
    SCHED_PHASE,
    canonical_phase,
    normalize_policy_action,
    normalize_policy_state,
)
from optimization.actions import stable_state
from optimization.decomposition_eligibility import workload_supports_decomposition
from training.state_builder import (
    STATE_ABLATIONS,
    ExecutionStateBuilder,
)

PHASE_LEVEL = {
    DEC_PHASE: "dec",
    SCHED_PHASE: "sched",
    ENUM_PHASE: "enum",
    ADAPT_PHASE: "adapt",
}

FROZEN_REPLAY_ENCODER_PHASES = frozenset({SCHED_PHASE, ENUM_PHASE, ADAPT_PHASE})

# Independent-Action pretraining may specialize the Adapt-only plan encoder.
# Sched and Enum still detach the shared query representation so their
# marginal fixed-profile labels cannot overwrite the Dec representation.
FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES = frozenset({SCHED_PHASE, ENUM_PHASE})

INITIAL_POLICY_PROFILES = (
    "postgres",
    "query_split",
    "top5",
    "lip_selective",
    "aja_conservative",
)


@dataclass
class ReplayTarget:
    """A fixed runtime-label target reconstructed from train-only experience."""

    state: StructuredState
    target: int
    mask: np.ndarray
    weight: float
    phase: str
    state_hash: str
    query_id: str
    action_costs_ms: dict[int, float]


def expand_legacy_checkpoint_tensors(
    model: HACNetwork, state_dict: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Migrate legacy action heads and the plan-only Adapt trunk."""
    return migrate_action_space_checkpoint_tensors(model, state_dict)


def load_model_checkpoint(
    path: Optional[Path],
    *,
    hidden: int,
    device: torch.device,
) -> tuple[HACNetwork, dict[str, Any], Optional[dict[str, Any]]]:
    model = HACNetwork(hidden=hidden).to(device)
    metadata: dict[str, Any] = {}
    optimizer_state = None
    if path is None:
        return model, metadata, optimizer_state

    try:
        # Online checkpoints are locally generated and include optimizer/RNG
        # metadata, so PyTorch 2.6's weights-only default cannot deserialize
        # them. Older PyTorch releases do not expose this keyword.
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and isinstance(payload.get("model_state"), dict):
        state_dict = payload["model_state"]
        metadata = dict(payload.get("metadata") or {})
        optimizer_state = payload.get("optimizer_state")
    elif isinstance(payload, dict) and isinstance(
        payload.get("model_state_dict"), dict
    ):
        state_dict = payload["model_state_dict"]
        metadata = dict(payload.get("metadata") or {})
        optimizer_state = payload.get("optimizer_state_dict")
    else:
        state_dict = payload
    state_dict, expanded = expand_legacy_checkpoint_tensors(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("sched_")]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={missing} unexpected={incompatible.unexpected_keys}"
        )
    if expanded:
        metadata["expanded_legacy_heads"] = expanded
        optimizer_state = None
    return model, metadata, optimizer_state


def _phase_action_index(phase: str, action: dict[str, Any]) -> int:
    phase = canonical_phase(phase)
    action = normalize_policy_action(action, phase=phase)
    if action.get("action_index") is not None:
        return int(action["action_index"])
    if phase == DEC_PHASE:
        return 1 if action.get("dec_action") == "apply" else 0
    if phase == SCHED_PHASE:
        if action.get("sched_idx") is not None:
            return int(action["sched_idx"])
        alpha = float(action.get("sched_alpha", 0.5))
        return min(
            range(N_SCHED),
            key=lambda index: abs(SCHED_ALPHA_VALUES[index] - alpha),
        )
    if phase == ENUM_PHASE:
        label = str(action.get("enum_action") or "native")
        # The learned Enum head intentionally represents the binary
        # Native-vs-TOP-K mechanism and currently executes TOP-5. Shared
        # experience can also contain standalone Top-10 calibration rows;
        # those rows are valid examples of the same enabled mechanism and
        # must not make replay/pretraining fail while decoding history.
        if label.startswith("top") and label not in ENUM_LABELS:
            label = "top5"
        return ENUM_LABELS.index(label)

    filter_action = str(action.get("filter_action") or "none")
    ajoin_action = str(action.get("ajoin_action") or "off")
    # Adapt states are captured before either mechanism is applied. Shared
    # experience can therefore contain standalone calibration variants
    # (LIP full and AJA aggressive/legacy ``aja``) even though the learned
    # Adapt head intentionally exposes only selective and conservative.  Map
    # those historical variants to the corresponding enabled mechanism for
    # state lookup; independent-action summaries below still provide the
    # exact selective/conservative training costs and labels.
    prefix = "filter_selective" if filter_action in {"selective", "full"} else ""
    if ajoin_action in {"conservative", "aggressive"}:
        label = f"{prefix}+ajoin_conservative" if prefix else "ajoin_conservative"
    else:
        label = prefix or "none"
    return ADAPT_LABELS.index(label)


def _phase_size(phase: str) -> int:
    phase = canonical_phase(phase)
    return {
        "dec": N_DEC,
        "sched": N_SCHED,
        "enum": N_ENUM,
        "adapt": N_ADAPT,
    }[phase]


def _decode_mask(raw: Any, phase: str) -> np.ndarray:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if raw is None:
        return np.ones(_phase_size(phase), dtype=np.float32)
    mask = np.asarray(raw, dtype=np.float32)
    expected = _phase_size(phase)
    if mask.size < expected:
        mask = np.pad(mask, (0, expected - mask.size))
    if mask.size != expected:
        raise ValueError(
            f"{phase} action mask has size {mask.size}, expected {expected}"
        )
    return mask


def _decision_rows(
    store: ExperienceStore,
    *,
    query_ids: Optional[Iterable[str]] = None,
    cutoff_ms: Optional[int] = None,
    policy_version: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Expand compact execution rows into in-memory decision samples."""
    rows: list[dict[str, Any]] = []
    for execution in store.iter_executions(
        query_ids=query_ids,
        cutoff_ms=cutoff_ms,
    ):
        round_events = {
            int(event.get("round") or 0): event
            for event in execution["db_events"]
            if isinstance(event, dict)
        }
        for position, decision in enumerate(execution["trajectory"]):
            if not isinstance(decision, dict):
                continue
            phase = canonical_phase(decision.get("phase"))
            if phase not in PHASE_LEVEL:
                continue
            state = normalize_policy_state(decision.get("state") or {})
            semantic_action = normalize_policy_action(
                decision.get("action") or {}, phase=phase
            )
            if not isinstance(state, dict) or not isinstance(semantic_action, dict):
                continue
            policy = decision.get("policy") or {}
            if not isinstance(policy, dict):
                policy = {}
            stored_policy_version = policy.get("policy_version")
            if policy_version is not None and stored_policy_version != policy_version:
                continue
            action = normalize_policy_action({**semantic_action, **policy}, phase=phase)
            round_index = int(decision.get("round_index", state.get("round") or 0))
            event = round_events.get(round_index, {})
            timing = event.get("timing_ms") or {}
            fallback_runtime = float(timing.get("total") or 0.0)
            if not fallback_runtime:
                fallback_runtime = float(timing.get("planning") or 0.0) + float(
                    timing.get("execution") or 0.0
                )
            is_timeout = bool(
                decision.get("is_timeout", execution["status"] == "timeout")
            )
            runtime_ms = decision.get("runtime_ms")
            if runtime_ms is None and not is_timeout:
                runtime_ms = fallback_runtime or execution["first_runtime_ms"]
            charged_runtime_ms = float(
                decision.get("charged_runtime_ms")
                or (execution["charged_runtime_ms"] if is_timeout else runtime_ms)
            )
            state_hash = str(
                decision.get("state_hash") or content_hash(stable_state(state))
            )
            implementation_version = str(
                decision.get("implementation_version") or "compact-v1"
            )
            action_hash = content_hash(semantic_action)
            rows.append(
                {
                    "decision_id": f"{execution['cache_id']}:{position}",
                    "episode_id": str(execution["source_episode_id"]),
                    "query_id": str(execution["query_id"]),
                    "round_index": round_index,
                    "phase": phase,
                    "state": state,
                    "state_hash": state_hash,
                    "state_blob_hash": content_hash(state),
                    "action": action,
                    "action_json": json.dumps(action),
                    "action_mask_json": policy.get("action_mask"),
                    "log_probability": policy.get("log_probability"),
                    "predicted_value": policy.get("predicted_value"),
                    "policy_version": stored_policy_version,
                    "implementation_version": implementation_version,
                    "measurement_key": content_hash(
                        {
                            "state": state_hash,
                            "action": action_hash,
                            "implementation": implementation_version,
                        }
                    ),
                    "downstream_signature": str(
                        decision.get("downstream_signature") or ""
                    ),
                    "runtime_ms": runtime_ms,
                    "charged_runtime_ms": charged_runtime_ms,
                    "training_runtime_ms": charged_runtime_ms,
                    "is_timeout": int(is_timeout),
                    "observed_at_ms": int(
                        decision.get("observed_at_ms") or execution["created_at_ms"]
                    ),
                    "started_at_ms": int(execution["created_at_ms"]),
                    "trajectory_hash": str(execution["trajectory_hash"]),
                    "episode_status": str(execution["status"]),
                }
            )
    return rows


def _replay_state_identity(phase: str, state: dict[str, Any]) -> Any:
    """Return the stable identity used to aggregate fixed runtime labels.

    Temporary-table ANALYZE sampling can change a baseline plan's estimated
    cost between otherwise identical residual states. Dec-level labels are
    therefore keyed by the residual query/relations, while the model input
    still retains the complete plan features from the selected state blob.
    """
    identity = stable_state(state)
    if phase != "dec" or not isinstance(identity, dict):
        return identity
    dynamic_plan_fields = {
        "plan_available",
        "plan_json",
        "plan_rows",
        "plan_state_ms",
        "plan_summary",
        "plan_total_cost",
        "plan_width",
    }
    return {
        key: value for key, value in identity.items() if key not in dynamic_plan_fields
    }


def collect_transitions(
    store: ExperienceStore,
    *,
    workload: str,
    policy_version: Optional[str],
    query_ids: Optional[Iterable[str]] = None,
    cutoff_ms: Optional[int] = None,
    require_stochastic: bool = True,
    reward_scale_ms: Optional[float] = None,
    reward_clip: float = 30.0,
    action_ablation: str = "none",
    state_ablation: str = "none",
    catalog: CatalogInfo | None = None,
) -> dict[str, list[Transition]]:
    rows = _decision_rows(
        store,
        query_ids=query_ids,
        cutoff_ms=cutoff_ms,
        policy_version=policy_version,
    )
    if cutoff_ms is not None:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(row["measurement_key"], row["downstream_signature"])].append(
                float(row["charged_runtime_ms"])
            )
        medians = {key: statistics.median(values) for key, values in grouped.items()}
        for row in rows:
            row["training_runtime_ms"] = medians[
                (row["measurement_key"], row["downstream_signature"])
            ]
    builder = ExecutionStateBuilder(workload, state_ablation, catalog=catalog)
    prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    sequential_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        action = row["action"]
        if require_stochastic and action.get("inference_mode") != "stochastic":
            continue
        state = row["state"]
        prepared.append((row, state, action))
        if row["phase"] in {"dec", "sched"}:
            sequential_counts[(row["episode_id"], row["phase"])] += 1

    if reward_scale_ms is None:
        episode_baselines = {
            str(row["episode_id"]): max(float(row["charged_runtime_ms"] or 1.0), 1.0)
            for row, _state, _action in prepared
        }
        reward_scale_ms = (
            statistics.mean(episode_baselines.values()) if episode_baselines else 1.0
        )
    reward_scale_ms = max(float(reward_scale_ms), 1.0)
    reward_clip = max(float(reward_clip), 1.0)

    sequential_seen: dict[tuple[str, str], int] = defaultdict(int)
    buffers: dict[str, list[Transition]] = {phase: [] for phase in PHASE_LEVEL}
    for row, state, action in prepared:
        phase = str(row["phase"])
        action_index = _phase_action_index(phase, action)
        if action_index >= _phase_size(phase):
            raise ValueError(f"{phase} action index {action_index} is outside its head")
        scaled_cost = float(row["training_runtime_ms"]) / reward_scale_ms
        reward = -min(scaled_cost, reward_clip)
        if phase in {"dec", "sched"}:
            sequence_key = (row["episode_id"], phase)
            sequential_seen[sequence_key] += 1
            done = sequential_seen[sequence_key] == sequential_counts[sequence_key]
        else:
            done = True
        structured = builder.build(phase, row["state_blob_hash"], state)
        mask = apply_action_ablation_mask(
            _decode_mask(row["action_mask_json"], phase),
            phase,
            action_ablation,
        )
        if mask[action_index] <= 0.0:
            raise ValueError(
                f"stored {phase} action {action_index} is disabled by "
                f"{action_ablation}"
            )
        raw_coverage_probs = action.get("coverage_probabilities")
        coverage_probs = None
        if isinstance(raw_coverage_probs, list):
            coverage_probs = np.asarray(raw_coverage_probs, dtype=np.float32)
            if coverage_probs.shape != mask.shape:
                raise ValueError(
                    f"stored {phase} coverage distribution has shape "
                    f"{coverage_probs.shape}, expected {mask.shape}"
                )
        buffers[phase].append(
            Transition(
                state=structured,
                action=action_index,
                reward=reward,
                done=done,
                log_prob=float(row["log_probability"] or 0.0),
                value=float(row["predicted_value"] or 0.0),
                mask=mask,
                duration=1,
                temperature=float(action.get("temperature", 1.0)),
                exploration_epsilon=float(action.get("exploration_epsilon", 0.0)),
                coverage_mix=float(action.get("coverage_mix", 0.0)),
                coverage_probs=coverage_probs,
            )
        )
    return buffers


def collect_replay_targets(
    store: ExperienceStore,
    *,
    workload: str,
    cutoff_ms: int,
    reward_scale_ms: Optional[float],
    query_ids: Iterable[str],
    minimum_samples: int = 1,
    action_ablation: str = "none",
    state_ablation: str = "none",
    catalog: CatalogInfo | None = None,
) -> dict[str, list[ReplayTarget]]:
    """Build counterfactual actor targets from fixed runtime labels.

    Every historical role, protocol, fold, and policy is eligible when its
    query belongs to the current fold's explicit training whitelist. Exact
    complete trajectories contribute only their first immutable measurement.
    Dec and Sched use Bellman returns over the observed residual-state graph:
    stop is terminal, while split costs the current round plus the learned
    value of the next residual state. Enum and Adapt use their current round
    runtime because they do not change query semantics. Default actions are
    learned from their own measured executions, just like every other action.
    """
    query_ids = sorted({str(query_id) for query_id in query_ids})
    if not query_ids:
        raise ValueError("runtime replay requires a nonempty training-query whitelist")
    rows = _decision_rows(store, query_ids=query_ids, cutoff_ms=cutoff_ms)
    first_episode_by_trajectory: dict[tuple[str, str], str] = {}
    deduplicated_rows: list[dict[str, Any]] = []
    for row in rows:
        episode_id = str(row["episode_id"])
        trajectory_identity = str(row["trajectory_hash"] or episode_id)
        key = (str(row["query_id"]), trajectory_identity)
        first_episode = first_episode_by_trajectory.setdefault(key, episode_id)
        if episode_id == first_episode:
            deduplicated_rows.append(row)
    rows = deduplicated_rows
    result: dict[str, list[ReplayTarget]] = {phase: [] for phase in PHASE_LEVEL}
    if not rows:
        return result
    if reward_scale_ms is None:
        reward_scale_ms = statistics.mean(
            max(float(row["charged_runtime_ms"]), 1.0) for row in rows
        )

    parsed: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (row, row["action"]) for row in rows
    ]
    action_by_round: dict[tuple[str, int, str], int] = {}
    round_cost: dict[tuple[str, int], float] = {}
    rounds_by_episode: dict[str, list[int]] = defaultdict(list)
    state_cache: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    dec_state_by_round: dict[tuple[str, int], str] = {}

    def state_entry(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
        blob_hash = str(row["state_blob_hash"])
        phase = str(row["phase"])
        cache_key = (phase, blob_hash)
        if cache_key not in state_cache:
            state = row["state"]
            state_cache[cache_key] = (
                state,
                content_hash(_replay_state_identity(phase, state)),
            )
        return state_cache[cache_key]

    for row, action in parsed:
        phase = str(row["phase"])
        episode_id = str(row["episode_id"])
        round_index = int(row["round_index"])
        action_by_round[(episode_id, round_index, phase)] = _phase_action_index(
            phase, action
        )
        if phase == "dec":
            round_key = (episode_id, round_index)
            round_cost[round_key] = float(row["charged_runtime_ms"])
            rounds_by_episode[episode_id].append(round_index)
            dec_state_by_round[round_key] = state_entry(row)[1]
    for episode_id in rounds_by_episode:
        rounds_by_episode[episode_id] = sorted(set(rounds_by_episode[episode_id]))

    group_info: dict[tuple[str, str], dict[str, Any]] = {}
    terminal_samples: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    transition_samples: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    ordinary_samples: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def register_group(
        row: dict[str, Any],
        phase: str,
        semantic_state_hash: str,
    ) -> tuple[str, str]:
        key = (phase, semantic_state_hash)
        mask = apply_action_ablation_mask(
            _decode_mask(row["action_mask_json"], phase),
            phase,
            action_ablation,
        )
        group = group_info.get(key)
        if group is None:
            group_info[key] = {
                "phase": phase,
                "state_hash": semantic_state_hash,
                "state_blob_hash": str(row["state_blob_hash"]),
                "state": row["state"],
                "query_id": str(row["query_id"]),
                "mask": mask,
            }
        else:
            group["mask"] = np.maximum(group["mask"], mask)
        return key

    for row, action in parsed:
        phase = str(row["phase"])
        episode_id = str(row["episode_id"])
        round_index = int(row["round_index"])
        action_index = _phase_action_index(phase, action)
        _state, semantic_state_hash = state_entry(row)
        group_key = register_group(row, phase, semantic_state_hash)
        dec_action = action_by_round.get((episode_id, round_index, "dec"))
        enum_action = action_by_round.get((episode_id, round_index, "enum"))
        adapt_action = action_by_round.get((episode_id, round_index, "adapt"))

        # Attribute runtime at one hierarchy level at a time. A Dec=Skip
        # trajectory that later chooses TOP-K or AJoin is not another sample
        # of the default Dec action; it is a different downstream policy.
        # Missing later decisions in focused unit tests mean their defaults.
        downstream_enum_is_native = enum_action in (None, 0)
        downstream_adapt_is_default = adapt_action in (None, 0)
        if phase in {"dec", "sched"} and (
            not downstream_enum_is_native or not downstream_adapt_is_default
        ):
            continue
        if phase == "enum" and not downstream_adapt_is_default:
            continue

        if phase in {"dec", "sched"}:
            current_cost = round_cost[(episode_id, round_index)]
            next_round = next(
                (
                    later_round
                    for later_round in rounds_by_episode[episode_id]
                    if later_round > round_index
                ),
                None,
            )
            if dec_action == 1 and next_round is not None:
                next_state_hash = dec_state_by_round[(episode_id, next_round)]
                transition_samples[group_key][action_index][next_state_hash].append(
                    current_cost
                )
            else:
                terminal_samples[group_key][action_index].append(current_cost)
        else:
            ordinary_samples[group_key][action_index].append(
                float(row["charged_runtime_ms"])
            )

    terminal_costs: dict[tuple[str, str], dict[int, float]] = {}
    for group_key, action_samples in terminal_samples.items():
        terminal_costs[group_key] = {
            int(action): float(statistics.median(samples))
            for action, samples in action_samples.items()
            if len(samples) >= minimum_samples
        }
    transition_costs: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for group_key, action_edges in transition_samples.items():
        aggregated: dict[int, dict[str, float]] = {}
        for action, destinations in action_edges.items():
            eligible = {
                destination: float(statistics.median(samples))
                for destination, samples in destinations.items()
                if len(samples) >= minimum_samples
            }
            if eligible:
                aggregated[int(action)] = eligible
        if aggregated:
            transition_costs[group_key] = aggregated

    def action_costs_with_value(
        phase: str,
        state_hash: str,
        dec_values: dict[str, float],
    ) -> dict[int, float]:
        key = (phase, state_hash)
        candidates: dict[int, list[float]] = defaultdict(list)
        for action, cost_ms in terminal_costs.get(key, {}).items():
            candidates[action].append(cost_ms)
        for action, destinations in transition_costs.get(key, {}).items():
            for next_state_hash, immediate_cost in destinations.items():
                if next_state_hash in dec_values:
                    candidates[action].append(
                        immediate_cost + dec_values[next_state_hash]
                    )
        mask = group_info.get(key, {}).get("mask")
        return {
            action: min(costs)
            for action, costs in candidates.items()
            if costs and mask is not None and action < len(mask) and mask[action] > 0.0
        }

    dec_state_hashes = {
        state_hash for phase, state_hash in group_info if phase == "dec"
    }
    dec_values: dict[str, float] = {}
    for _pass in range(len(dec_state_hashes) + 1):
        changed = False
        for state_hash in dec_state_hashes:
            action_costs = action_costs_with_value("dec", state_hash, dec_values)
            if not action_costs:
                continue
            value = min(action_costs.values())
            if state_hash not in dec_values or not math.isclose(
                dec_values[state_hash],
                value,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                dec_values[state_hash] = value
                changed = True
        if not changed:
            break

    groups: list[dict[str, Any]] = []
    for (phase, state_hash), info in group_info.items():
        if phase in {"dec", "sched"}:
            action_costs = action_costs_with_value(phase, state_hash, dec_values)
        else:
            action_costs = {
                int(action): float(statistics.median(samples))
                for action, samples in ordinary_samples[(phase, state_hash)].items()
                if len(samples) >= minimum_samples
            }
        if action_costs:
            groups.append({**info, "action_costs": action_costs})

    builder = ExecutionStateBuilder(workload, state_ablation, catalog=catalog)
    for group in groups:
        mask = group["mask"]
        action_costs = {
            action: cost
            for action, cost in group["action_costs"].items()
            if action < len(mask) and mask[action] > 0.0
        }
        if len(action_costs) < 2:
            continue
        target = min(
            action_costs,
            key=lambda action: (action_costs[action], action),
        )
        if target >= len(mask) or mask[target] <= 0.0:
            continue

        ordered_costs = sorted(action_costs.values())
        gap_ms = max(ordered_costs[1] - ordered_costs[0], 0.0)
        weight = min(
            10.0,
            max(gap_ms / max(float(reward_scale_ms), 1.0), 0.05),
        )
        state = group["state"]
        result[group["phase"]].append(
            ReplayTarget(
                state=builder.build(
                    group["phase"],
                    group["state_blob_hash"],
                    state,
                ),
                target=target,
                mask=mask,
                weight=weight,
                phase=group["phase"],
                state_hash=group["state_hash"],
                query_id=group["query_id"],
                action_costs_ms=action_costs,
            )
        )
    return result


def collect_independent_action_targets(
    store: ExperienceStore,
    *,
    workload: str,
    summary_path: Path,
    reward_scale_ms: float,
    query_ids: Iterable[str],
    action_ablation: str = "none",
    prior_mode: str = "direct",
    crossfit_confidence_z: float = 1.645,
    crossfit_trees: int = 100,
    seed: int = 42,
    state_ablation: str = "none",
    catalog: CatalogInfo | None = None,
) -> dict[str, list[ReplayTarget]]:
    """Build train-only root-state priors from independent Action results.

    Fixed-profile experiments predate policy decision logging, so their exact
    runtimes live in summary.json while compatible model states come from
    later online episodes.  The query whitelist is mandatory: held-out
    independent-action outcomes never become training labels.
    """
    query_ids = sorted({str(query_id) for query_id in query_ids})
    if not query_ids:
        raise ValueError("independent-action prior requires training queries")
    if prior_mode not in {"direct", "conservative_crossfit"}:
        raise ValueError(f"unknown independent prior mode {prior_mode!r}")
    if crossfit_confidence_z < 0.0:
        raise ValueError("crossfit confidence z must be nonnegative")
    if crossfit_trees < 1:
        raise ValueError("crossfit trees must be positive")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_profiles = (
        ("query_split", "top5", "lip_selective", "aja_conservative")
        if workload_supports_decomposition(workload)
        else ("top5", "lip_selective", "aja_conservative")
    )
    for profile in required_profiles:
        if not isinstance(summary.get(profile), dict):
            raise ValueError(
                f"independent-action summary is missing profile {profile!r}"
            )
        missing = sorted(set(query_ids) - set(summary[profile].get("queries") or {}))
        if missing:
            raise ValueError(
                f"independent-action profile {profile!r} is missing "
                f"{len(missing)} training queries"
            )

    rows = [
        row
        for row in reversed(_decision_rows(store, query_ids=query_ids))
        if int(row["round_index"]) == 0
    ]
    result: dict[str, list[ReplayTarget]] = {phase: [] for phase in PHASE_LEVEL}
    if not rows:
        return result

    episode_actions: dict[tuple[str, str], int] = {}
    for row in rows:
        episode_actions[(str(row["episode_id"]), str(row["phase"]))] = (
            _phase_action_index(str(row["phase"]), row["action"])
        )

    def suitable(row: dict[str, Any]) -> bool:
        episode_id = str(row["episode_id"])
        phase = str(row["phase"])
        dec = episode_actions.get((episode_id, "dec"))
        enum = episode_actions.get((episode_id, "enum"))
        if phase == "sched":
            return dec == 1
        if phase == "enum":
            return dec == 0
        if phase == "adapt":
            return dec == 0 and enum == 0
        return phase == "dec"

    selected_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["query_id"]), str(row["phase"]))
        if key not in selected_rows and suitable(row):
            selected_rows[key] = row

    def profile_cost(profile: str, query_id: str) -> float:
        return float(summary[profile]["queries"][query_id]["median_charged_ms"])

    def postgres_cost(query_id: str) -> float:
        estimates = []
        for profile in required_profiles:
            speedup = (summary[profile].get("per_query_speedups") or {}).get(query_id)
            if speedup is not None:
                estimates.append(profile_cost(profile, query_id) * float(speedup))
        if not estimates:
            raise ValueError(
                "independent-action summary lacks PG-derived speedups for "
                f"{query_id}"
            )
        return float(statistics.median(estimates))

    builder = ExecutionStateBuilder(workload, state_ablation, catalog=catalog)
    for query_id in query_ids:
        pg_ms = max(postgres_cost(query_id), 1e-6)
        phase_costs: dict[str, dict[int, float]] = {
            "enum": {0: pg_ms, 1: profile_cost("top5", query_id)},
            "adapt": {
                0: pg_ms,
                ADAPT_LABELS.index("filter_selective"): profile_cost(
                    "lip_selective", query_id
                ),
                ADAPT_LABELS.index("ajoin_conservative"): profile_cost(
                    "aja_conservative", query_id
                ),
            },
        }
        if "query_split" in required_profiles:
            phase_costs["dec"] = {
                0: pg_ms,
                1: profile_cost("query_split", query_id),
            }
        # Alpha=0.5 is the only independently measured Sched action. It is
        # a useful prior only where that complete split policy beats default.
        if "dec" in phase_costs and phase_costs["dec"][1] < phase_costs["dec"][0]:
            phase_costs["sched"] = {
                SCHED_ALPHA_VALUES.index(0.5): phase_costs["dec"][1]
            }

        for phase, action_costs in phase_costs.items():
            row = selected_rows.get((query_id, phase))
            if row is None:
                continue
            mask = apply_action_ablation_mask(
                _decode_mask(row["action_mask_json"], phase),
                phase,
                action_ablation,
            )
            available_costs = {
                action: cost
                for action, cost in action_costs.items()
                if action < len(mask) and mask[action] > 0.0
            }
            if not available_costs:
                continue
            target = min(
                available_costs,
                key=lambda action: (available_costs[action], action),
            )
            ordered = sorted(available_costs.values())
            if len(ordered) >= 2:
                gap_ms = max(ordered[1] - ordered[0], 0.0)
            else:
                gap_ms = max(pg_ms - ordered[0], 0.0)
            weight = min(
                10.0,
                max(gap_ms / max(float(reward_scale_ms), 1.0), 0.05),
            )
            state = row["state"]
            result[phase].append(
                ReplayTarget(
                    state=builder.build(
                        phase,
                        str(row["state_blob_hash"]),
                        state,
                    ),
                    target=target,
                    mask=mask,
                    weight=weight,
                    phase=phase,
                    state_hash=str(row["state_hash"]),
                    query_id=query_id,
                    action_costs_ms=available_costs,
                )
            )
    if prior_mode == "conservative_crossfit":
        _apply_conservative_crossfit_prior(
            result,
            summary=summary,
            query_ids=query_ids,
            workload=workload,
            postgres_cost=postgres_cost,
            confidence_z=crossfit_confidence_z,
            trees=crossfit_trees,
            seed=seed,
        )
    return result


def collect_residual_split_prior_targets(
    store: ExperienceStore,
    *,
    workload: str,
    query_ids: list[str],
    action_ablation: str = "none",
    state_ablation: str = "none",
    catalog: CatalogInfo | None = None,
) -> list[ReplayTarget]:
    """Build train-only query-split priors for observed residual Dec states.

    The independently measured query-split profile supplies more than a root
    decision: after each materialization it chooses split again while another
    legal split exists.  Reusing those observed residual states as a prior
    prevents root cost fitting from forgetting the continuation behavior.  A
    state is deduplicated within each training query, and test-query states are
    excluded by the explicit whitelist.
    """
    query_ids = sorted(set(query_ids))
    if not query_ids:
        return []
    rows = [
        row
        for row in reversed(_decision_rows(store, query_ids=query_ids))
        if row["phase"] == "dec" and int(row["round_index"]) > 0
    ]
    builder = ExecutionStateBuilder(workload, state_ablation, catalog=catalog)
    seen: set[tuple[str, str]] = set()
    targets: list[ReplayTarget] = []
    for row in rows:
        query_id = str(row["query_id"])
        state_hash = str(row["state_hash"])
        key = (query_id, state_hash)
        if key in seen:
            continue
        seen.add(key)
        mask = apply_action_ablation_mask(
            _decode_mask(row["action_mask_json"], "dec"),
            "dec",
            action_ablation,
        )
        if len(mask) < 2 or mask[1] <= 0.0:
            continue
        state_blob_hash = str(row["state_blob_hash"])
        targets.append(
            ReplayTarget(
                state=builder.build(
                    "dec",
                    state_blob_hash,
                    row["state"],
                ),
                target=1,
                mask=mask,
                weight=1.0,
                phase="dec",
                state_hash=state_hash,
                query_id=query_id,
                action_costs_ms={1: 1.0},
            )
        )
    return targets


def _pooled_array_features(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    width = values.shape[1] if values.ndim == 2 else 1
    if values.shape[0] == 0:
        return np.zeros(1 + 4 * width, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray([values.shape[0]], dtype=np.float32),
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
        ]
    ).astype(np.float32)


def _independent_query_features(state: StructuredState) -> np.ndarray:
    graph = state.query_graph
    return np.concatenate(
        [
            _pooled_array_features(graph.table_node_features),
            _pooled_array_features(graph.column_node_features),
            _pooled_array_features(graph.join_edge_features),
            _pooled_array_features(graph.table_join_edge_features),
            np.asarray(state.ctx, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def _apply_conservative_crossfit_prior(
    targets: dict[str, list[ReplayTarget]],
    *,
    summary: dict[str, Any],
    query_ids: list[str],
    workload: str,
    postgres_cost: Any,
    confidence_z: float,
    trees: int,
    seed: int,
) -> None:
    """Replace direct labels with train-only, family-cross-fitted safe labels.

    A runtime ensemble predicts all independent profiles while holding out one
    query family at a time.  The training-selected best global profile is the
    reference.  A different profile becomes a pseudo-label only when its
    upper runtime bound is below the reference's lower bound.  This keeps
    unseen-family behavior conservative without a policy-probability cutoff.
    """
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as exc:  # pragma: no cover - deployment validation
        raise RuntimeError("conservative crossfit priors require scikit-learn") from exc

    profiles = (
        "postgres",
        "query_split",
        "top5",
        "lip_selective",
        "aja_conservative",
    )
    dec_by_query = {target.query_id: target for target in targets["dec"]}
    missing = sorted(set(query_ids) - set(dec_by_query))
    if missing:
        raise RuntimeError(
            f"conservative crossfit prior lacks Dec states for {len(missing)} queries"
        )

    features = np.stack(
        [
            _independent_query_features(dec_by_query[query_id].state)
            for query_id in query_ids
        ]
    )
    costs = []
    for query_id in query_ids:
        pg_ms = max(
            float(postgres_cost(query_id, 1.0)),
            1e-6,
        )
        costs.append(
            [
                pg_ms,
                *[
                    float(summary[profile]["queries"][query_id]["median_charged_ms"])
                    for profile in profiles[1:]
                ],
            ]
        )
    costs_array = np.asarray(costs, dtype=np.float64)
    log_relative_costs = np.log(np.maximum(costs_array / costs_array[:, :1], 1e-6))
    reference = min(
        range(len(profiles)),
        key=lambda index: (float(costs_array[:, index].sum()), index),
    )
    families: dict[str, list[int]] = defaultdict(list)
    for index, query_id in enumerate(query_ids):
        families[_base_query_family(query_id, workload=workload)].append(index)

    chosen_profiles = np.full(len(query_ids), reference, dtype=np.int64)
    confidence_margins = np.zeros(len(query_ids), dtype=np.float64)
    all_indices = np.arange(len(query_ids), dtype=np.int64)
    for family_index, validation_indices in enumerate(families.values()):
        validation = np.asarray(validation_indices, dtype=np.int64)
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        if len(training) < 2:
            continue
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed + family_index,
            n_jobs=1,
        )
        model.fit(features[training], log_relative_costs[training])
        tree_predictions = np.stack(
            [tree.predict(features[validation]) for tree in model.estimators_]
        )
        mean = tree_predictions.mean(axis=0)
        deviation = tree_predictions.std(axis=0)
        upper = mean + confidence_z * deviation
        reference_lower = mean[:, reference] - confidence_z * deviation[:, reference]
        eligible = upper < reference_lower[:, None]
        eligible[:, reference] = True
        conservative_scores = np.where(eligible, upper, np.inf)
        selected = conservative_scores.argmin(axis=1)
        chosen_profiles[validation] = selected
        confidence_margins[validation] = np.maximum(
            reference_lower - conservative_scores[np.arange(len(validation)), selected],
            0.0,
        )

    dec_actions = {
        "postgres": 0,
        "query_split": 1,
        "top5": 0,
        "lip_selective": 0,
        "aja_conservative": 0,
    }
    enum_actions = {
        "postgres": 0,
        "query_split": 0,
        "top5": ENUM_LABELS.index("top5"),
        "lip_selective": 0,
        "aja_conservative": 0,
    }
    adapt_actions = {
        "postgres": 0,
        "query_split": 0,
        "top5": 0,
        "lip_selective": ADAPT_LABELS.index("filter_selective"),
        "aja_conservative": ADAPT_LABELS.index("ajoin_conservative"),
    }
    phase_actions = {
        "dec": dec_actions,
        "enum": enum_actions,
        "adapt": adapt_actions,
    }
    profile_by_query = {
        query_id: profiles[int(chosen_profiles[index])]
        for index, query_id in enumerate(query_ids)
    }
    margin_by_query = {
        query_id: float(confidence_margins[index])
        for index, query_id in enumerate(query_ids)
    }
    for phase, action_map in phase_actions.items():
        for target in targets[phase]:
            profile = profile_by_query[target.query_id]
            selected_action = action_map[profile]
            if (
                selected_action < len(target.mask)
                and target.mask[selected_action] > 0.0
            ):
                target.target = selected_action
            target.weight = max(1.0 + margin_by_query[target.query_id], 0.05)


def replay_policy_update(
    model: HACNetwork,
    optimizer: torch.optim.Optimizer,
    targets: list[ReplayTarget],
    level: str,
    device: torch.device,
    *,
    epochs: int = 4,
    batch_size: int = 64,
    importance_power: float = 0.0,
    freeze_encoder: bool = False,
    sched_cost_temperature: float = 0.1,
    action_cost_temperature: Optional[float] = None,
    action_cost_regression: bool = False,
    balance_classes: Optional[bool] = None,
) -> float:
    """Fit an actor head to the best fixed runtime label per semantic state."""
    level = canonical_phase(level)
    if not targets or epochs <= 0:
        return 0.0
    if action_cost_regression and action_cost_temperature is not None:
        raise ValueError(
            "action cost regression and soft cost targets are mutually exclusive"
        )
    sample_weights = _balanced_replay_weights(
        targets,
        importance_power=importance_power,
        # Dec is a dense binary decision and benefits from class balancing.
        # Enum and Adapt have sparse, workload-specific wins; balancing their
        # rare actions would make a handful of local improvements dominate the
        # PostgreSQL-safe default class.
        balance_classes=(
            level == "dec" if balance_classes is None else balance_classes
        ),
        balance_scopes=(
            level == "dec"
            if balance_classes is None
            else balance_classes and level == "dec"
        ),
    )
    model.train()
    total_loss = 0.0
    updates = 0
    for _epoch in range(epochs):
        permutation = np.random.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            indices = permutation[start : start + batch_size]
            train_private_trunk = freeze_encoder and level in {"enum", "adapt"}
            hidden = model.encode_batch(
                targets,
                indices,
                device,
                detach_shared=train_private_trunk,
            )
            if freeze_encoder and not train_private_trunk:
                hidden = hidden.detach()
            if level == "dec":
                logits = model.dec_actor(hidden)
            elif level == "sched":
                logits = model.sched_actor(model.sched_features(hidden))
            elif level == "enum":
                logits = model.enum_actor(hidden)
            else:
                logits = model.adapt_actor(hidden)
            masks = torch.stack(
                [
                    torch.from_numpy(targets[index].mask).float().to(device)
                    for index in indices
                ]
            )
            logits = logits + (masks - 1.0) * 1e9
            labels = torch.tensor(
                [targets[index].target for index in indices],
                dtype=torch.long,
                device=device,
            )
            weights = torch.tensor(
                [sample_weights[index] for index in indices],
                dtype=torch.float32,
                device=device,
            )
            if action_cost_regression:
                regression_targets, measured_masks = zip(
                    *[
                        _action_cost_regression_target(
                            targets[index],
                            device,
                            size=logits.shape[-1],
                        )
                        for index in indices
                    ]
                )
                regression_targets = torch.stack(regression_targets)
                measured_masks = torch.stack(measured_masks)
                losses = (((logits - regression_targets) ** 2) * measured_masks).sum(
                    dim=-1
                ) / measured_masks.sum(dim=-1).clamp_min(1.0)
            elif action_cost_temperature is not None:
                soft_targets = torch.stack(
                    [
                        _action_cost_target(
                            targets[index],
                            device,
                            size=logits.shape[-1],
                            temperature=action_cost_temperature,
                        )
                        for index in indices
                    ]
                )
                losses = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            elif level == "sched":
                soft_targets = torch.stack(
                    [
                        _sched_cost_target(
                            targets[index],
                            device,
                            temperature=sched_cost_temperature,
                        )
                        for index in indices
                    ]
                )
                losses = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            else:
                losses = F.cross_entropy(logits, labels, reduction="none")
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            updates += 1
    return total_loss / max(updates, 1)


def joint_dec_prior_update(
    model: HACNetwork,
    optimizer: torch.optim.Optimizer,
    root_targets: list[ReplayTarget],
    residual_targets: list[ReplayTarget],
    device: torch.device,
    *,
    epochs: int,
    root_cost_regression: bool = True,
) -> dict[str, float]:
    """Jointly retain root costs and query-split continuation behavior."""
    if epochs <= 0 or not root_targets or not residual_targets:
        return {"root_cost_regression": 0.0, "residual_split": 0.0}
    root_indices = np.arange(len(root_targets))
    residual_indices = np.arange(len(residual_targets))
    totals = {"root_cost_regression": 0.0, "residual_split": 0.0}
    model.train()
    for _epoch in range(epochs):
        root_hidden = model.encode_batch(root_targets, root_indices, device)
        root_logits = model.dec_actor(root_hidden)
        root_masks = torch.tensor(
            np.stack([target.mask for target in root_targets]),
            dtype=torch.float32,
            device=device,
        )
        root_logits = root_logits + (root_masks - 1.0) * 1e9
        if root_cost_regression:
            regression_targets, measured_masks = zip(
                *[
                    _action_cost_regression_target(
                        target,
                        device,
                        size=root_logits.shape[-1],
                    )
                    for target in root_targets
                ]
            )
            regression_targets_tensor = torch.stack(regression_targets)
            measured_masks_tensor = torch.stack(measured_masks)
            root_loss = (
                ((root_logits - regression_targets_tensor) ** 2) * measured_masks_tensor
            ).sum(dim=-1) / measured_masks_tensor.sum(dim=-1).clamp_min(1.0)
            root_loss = root_loss.mean()
        else:
            root_labels = torch.tensor(
                [target.target for target in root_targets],
                dtype=torch.long,
                device=device,
            )
            root_loss = F.cross_entropy(root_logits, root_labels)

        residual_hidden = model.encode_batch(
            residual_targets,
            residual_indices,
            device,
        )
        residual_logits = model.dec_actor(residual_hidden)
        residual_masks = torch.tensor(
            np.stack([target.mask for target in residual_targets]),
            dtype=torch.float32,
            device=device,
        )
        residual_logits = residual_logits + (residual_masks - 1.0) * 1e9
        residual_labels = torch.ones(
            len(residual_targets),
            dtype=torch.long,
            device=device,
        )
        residual_loss = F.cross_entropy(residual_logits, residual_labels)

        optimizer.zero_grad()
        (root_loss + residual_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["root_cost_regression"] += float(root_loss.item())
        totals["residual_split"] += float(residual_loss.item())
    return {key: value / epochs for key, value in totals.items()}


def _balanced_replay_weights(
    targets: list[ReplayTarget],
    *,
    importance_power: float,
    balance_classes: bool = True,
    balance_scopes: bool = False,
) -> np.ndarray:
    """Balance target classes while optionally preserving within-class impact."""
    if importance_power < 0.0:
        raise ValueError("importance_power must be nonnegative")
    weights = np.asarray(
        [max(float(target.weight), 1e-6) ** importance_power for target in targets],
        dtype=np.float32,
    )
    classes = sorted({target.target for target in targets})
    if not balance_classes:
        return weights
    scopes = (
        sorted({_dec_replay_scope(target) for target in targets})
        if balance_scopes
        else [None]
    )
    for scope in scopes:
        scoped_classes = {
            target.target
            for target in targets
            if scope is None or _dec_replay_scope(target) == scope
        }
        if len(scoped_classes) < 2 and not balance_scopes:
            continue
        for action in sorted(scoped_classes):
            indices = np.asarray(
                [
                    index
                    for index, target in enumerate(targets)
                    if target.target == action
                    and (scope is None or _dec_replay_scope(target) == scope)
                ],
                dtype=np.int64,
            )
            class_total = float(weights[indices].sum())
            if class_total > 0.0:
                weights[indices] /= class_total
    return weights


def _dec_replay_scope(target: ReplayTarget) -> str:
    ctx = np.asarray(target.state.ctx, dtype=np.float32).reshape(-1)
    return "residual" if len(ctx) > 1 and float(ctx[1]) > 0.0 else "root"


def _sched_cost_target(
    target: ReplayTarget,
    device: torch.device,
    *,
    temperature: float,
) -> torch.Tensor:
    """Convert noisy alpha runtimes into a relative-regret distribution."""
    if temperature <= 0.0:
        raise ValueError("Sched cost temperature must be positive")
    costs = torch.full(
        (N_SCHED,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    for action, cost_ms in target.action_costs_ms.items():
        if 0 <= action < N_SCHED and target.mask[action] > 0.0:
            costs[action] = max(float(cost_ms), 0.0)
    available = torch.isfinite(costs)
    if not bool(available.any()):
        raise ValueError("Sched replay target has no available costs")
    best = costs[available].min().clamp_min(1.0)
    relative_regret = (costs - best) / best
    logits = -relative_regret / temperature
    logits = torch.where(
        available,
        logits,
        torch.full_like(logits, -1e9),
    )
    return torch.softmax(logits, dim=-1)


def _action_cost_target(
    target: ReplayTarget,
    device: torch.device,
    *,
    size: int,
    temperature: float,
) -> torch.Tensor:
    """Turn measured per-action regret into a cost-sensitive soft label."""
    if temperature <= 0.0:
        raise ValueError("action cost temperature must be positive")
    costs = torch.full(
        (size,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    for action, cost_ms in target.action_costs_ms.items():
        if 0 <= action < size and target.mask[action] > 0.0:
            costs[action] = max(float(cost_ms), 0.0)
    available = torch.isfinite(costs)
    if not bool(available.any()):
        raise ValueError("replay target has no measured available actions")
    best = costs[available].min().clamp_min(1.0)
    relative_regret = (costs - best) / best
    logits = torch.where(
        available,
        -relative_regret / temperature,
        torch.full_like(relative_regret, -1e9),
    )
    return torch.softmax(logits, dim=-1)


def _action_cost_regression_target(
    target: ReplayTarget,
    device: torch.device,
    *,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-relative runtime scores and their measured-action mask.

    The best measured action has score zero; slower actions have negative
    scores.  Fitting actor logits to these scores preserves runtime magnitude
    while deterministic inference still selects the minimum predicted cost.
    """
    costs = torch.full(
        (size,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    for action, cost_ms in target.action_costs_ms.items():
        if 0 <= action < size and target.mask[action] > 0.0:
            costs[action] = max(float(cost_ms), 1e-6)
    measured = torch.isfinite(costs)
    if not bool(measured.any()):
        raise ValueError("replay target has no measured available actions")
    best = costs[measured].min().clamp_min(1e-6)
    scores = torch.zeros_like(costs)
    scores[measured] = -torch.log(costs[measured] / best).clamp(max=5.0)
    return scores, measured.float()


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def portable_numpy_rng_state() -> dict[str, Any]:
    """Serialize NumPy RNG state without NumPy-version-specific classes."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(name),
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def initialize_policy_profile(
    model: HACNetwork,
    *,
    profile: str,
    sched_alpha: float,
    action_bias: float = 0.5,
) -> None:
    """Start exploration from one measured independent-action policy."""
    if profile not in INITIAL_POLICY_PROFILES:
        raise ValueError(
            f"unknown initial policy profile {profile!r}; "
            f"expected one of {INITIAL_POLICY_PROFILES}"
        )
    sched_index = min(
        range(N_SCHED),
        key=lambda index: abs(SCHED_ALPHA_VALUES[index] - sched_alpha),
    )
    selected_actions = {
        "postgres": (0, sched_index, 0, 0),
        "query_split": (1, sched_index, 0, 0),
        "top5": (0, sched_index, ENUM_LABELS.index("top5"), 0),
        "lip_selective": (
            0,
            sched_index,
            0,
            ADAPT_LABELS.index("filter_selective"),
        ),
        "aja_conservative": (
            0,
            sched_index,
            0,
            ADAPT_LABELS.index("ajoin_conservative"),
        ),
    }[profile]
    with torch.no_grad():
        for head, selected in (
            (model.dec_actor, selected_actions[0]),
            (model.sched_actor, selected_actions[1]),
            (model.enum_actor, selected_actions[2]),
            (model.adapt_actor, selected_actions[3]),
        ):
            head.weight.zero_()
            head.bias.zero_()
            head.bias[selected] = action_bias
        for critic in (
            model.dec_critic,
            model.sched_critic,
            model.enum_critic,
            model.adapt_critic,
        ):
            critic.weight.zero_()
            critic.bias.zero_()


def initialize_safe_policy(
    model: HACNetwork,
    *,
    sched_alpha: float,
    action_bias: float = 0.5,
) -> None:
    """Backward-compatible PostgreSQL-equivalent initialization."""
    initialize_policy_profile(
        model,
        profile="postgres",
        sched_alpha=sched_alpha,
        action_bias=action_bias,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-db", type=Path, required=True)
    parser.add_argument("--workload", choices=("job", "stack", "tpch"), required=True)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        help="database-derived catalog snapshot created at run startup",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help=(
            "skip on-policy PPO transitions and train only from the fixed "
            "query-whitelisted replay groups"
        ),
    )
    parser.add_argument(
        "--source-policy-version",
        help="only train on decisions generated by this policy version",
    )
    parser.add_argument(
        "--policy-version",
        help="override the version assigned to the new checkpoint",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-gae", type=float, default=0.95)
    parser.add_argument(
        "--initial-sched-alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument("--initial-action-bias", type=float, default=0.5)
    parser.add_argument(
        "--initial-policy-profile",
        choices=INITIAL_POLICY_PROFILES,
        default="postgres",
    )
    parser.add_argument(
        "--reward-scale-ms",
        type=float,
        help=(
            "workload-wide runtime scale; a shared scale preserves the "
            "sum-runtime objective used by workload speedup"
        ),
    )
    parser.add_argument("--reward-clip", type=float, default=30.0)
    parser.add_argument(
        "--action-ablation",
        choices=ACTION_ABLATIONS,
        default="none",
        help="mask one optimizer mechanism during PPO and runtime replay",
    )
    parser.add_argument(
        "--state-ablation",
        choices=STATE_ABLATIONS,
        default="none",
        help="remove query-graph or plan-tree topology while retaining features",
    )
    parser.add_argument("--split-protocol")
    parser.add_argument("--fold")
    parser.add_argument(
        "--replay-query-id",
        action="append",
        help=(
            "training-query whitelist for fixed runtime replay; repeat for "
            "multiple queries"
        ),
    )
    parser.add_argument(
        "--replay-group",
        action="append",
        help=(
            "only build fixed runtime targets from these counterfactual "
            "groups; repeat for multiple compatible groups"
        ),
    )
    parser.add_argument("--replay-epochs", type=int, default=16)
    parser.add_argument("--dec-replay-epochs", type=int)
    parser.add_argument("--sched-replay-epochs", type=int)
    parser.add_argument("--enum-replay-epochs", type=int)
    parser.add_argument("--adapt-replay-epochs", type=int)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--replay-minimum-samples", type=int, default=1)
    parser.add_argument(
        "--replay-importance-power",
        type=float,
        default=0.5,
        help=(
            "within-class runtime-gap weighting exponent; zero gives each "
            "fixed label equal weight while balancing action classes"
        ),
    )
    for phase in ("dec", "sched", "enum", "adapt"):
        parser.add_argument(
            f"--{phase}-replay-importance-power",
            type=float,
            help=(
                f"override --replay-importance-power for the {phase} " "replay update"
            ),
        )
    parser.add_argument(
        "--sched-replay-temperature",
        type=float,
        default=0.1,
        help=("relative-runtime temperature for soft alpha replay targets"),
    )
    parser.add_argument(
        "--replay-action-cost-temperature",
        type=float,
        default=0.0,
        help=(
            "positive relative-regret temperature enables cost-sensitive "
            "soft targets for Dec, Enum, and Adapt runtime replay; zero "
            "keeps hard runtime labels"
        ),
    )
    parser.add_argument(
        "--replay-action-cost-regression",
        action="store_true",
        help=(
            "fit actor logits to measured log-relative runtimes for Dec, "
            "Enum, and Adapt runtime replay"
        ),
    )
    parser.add_argument(
        "--disable-runtime-replay",
        action="store_true",
        help="disable fixed counterfactual runtime-label actor updates",
    )
    parser.add_argument(
        "--independent-action-summary",
        type=Path,
        help=(
            "fixed-profile runtime summary used for train-whitelisted "
            "independent Action priors"
        ),
    )
    parser.add_argument(
        "--independent-prior-epochs",
        type=int,
        default=0,
    )
    for phase in ("dec", "sched", "enum", "adapt"):
        parser.add_argument(f"--independent-{phase}-prior-epochs", type=int)
    parser.add_argument(
        "--independent-prior-temperature",
        type=float,
        default=0.0,
        help=(
            "positive relative-regret temperature enables cost-sensitive "
            "soft labels for independent Action priors; zero keeps hard labels"
        ),
    )
    parser.add_argument(
        "--independent-prior-cost-regression",
        action="store_true",
        help=(
            "fit actor logits to train-only independent Action runtimes "
            "instead of hard or soft action labels"
        ),
    )
    parser.add_argument(
        "--residual-split-prior-epochs",
        type=int,
        default=0,
        help=(
            "jointly retain root independent-action costs and train-only "
            "query-split continuation states for this many bootstrap epochs"
        ),
    )
    parser.add_argument(
        "--residual-split-root-objective",
        choices=("cost_regression", "hard_label"),
        default="cost_regression",
    )
    parser.add_argument(
        "--independent-prior-mode",
        choices=("direct", "conservative_crossfit"),
        default="direct",
    )
    parser.add_argument(
        "--independent-crossfit-confidence-z", type=float, default=1.645
    )
    parser.add_argument("--independent-crossfit-trees", type=int, default=100)
    parser.add_argument(
        "--random-initial-policy",
        action="store_true",
        help="do not bias a fresh checkpoint toward PostgreSQL-safe actions",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.catalog_path is None:
        parser.error("--catalog-path is required")
    if not args.catalog_path.is_file():
        parser.error(f"catalog snapshot does not exist: {args.catalog_path}")
    catalog = CatalogInfo(args.catalog_path.resolve())
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if not 0.0 <= args.initial_sched_alpha <= 1.0:
        parser.error("--initial-sched-alpha must be in [0, 1]")
    if args.initial_action_bias < 0.0:
        parser.error("--initial-action-bias must be nonnegative")
    if args.reward_scale_ms is not None and args.reward_scale_ms <= 0.0:
        parser.error("--reward-scale-ms must be positive")
    if args.reward_clip <= 0.0:
        parser.error("--reward-clip must be positive")
    if args.replay_epochs < 0:
        parser.error("--replay-epochs must be nonnegative")
    phase_replay_epochs = {
        "dec": args.dec_replay_epochs,
        "sched": args.sched_replay_epochs,
        "enum": args.enum_replay_epochs,
        "adapt": args.adapt_replay_epochs,
    }
    for phase, configured_epochs in phase_replay_epochs.items():
        if configured_epochs is not None and configured_epochs < 0:
            parser.error(f"--{phase}-replay-epochs must be nonnegative")
        phase_replay_epochs[phase] = (
            args.replay_epochs if configured_epochs is None else configured_epochs
        )
    phase_replay_importance_power = {
        phase: getattr(args, f"{phase}_replay_importance_power")
        for phase in PHASE_LEVEL
    }
    for phase, configured_power in phase_replay_importance_power.items():
        if configured_power is not None and configured_power < 0.0:
            parser.error(f"--{phase}-replay-importance-power must be nonnegative")
        phase_replay_importance_power[phase] = (
            args.replay_importance_power
            if configured_power is None
            else configured_power
        )
    runtime_replay_enabled = not args.disable_runtime_replay and any(
        int(value) > 0 for value in phase_replay_epochs.values()
    )
    if args.independent_prior_epochs < 0:
        parser.error("--independent-prior-epochs must be nonnegative")
    independent_phase_epochs = {
        phase: getattr(args, f"independent_{phase}_prior_epochs")
        for phase in PHASE_LEVEL
    }
    for phase, configured_epochs in independent_phase_epochs.items():
        if configured_epochs is not None and configured_epochs < 0:
            parser.error(f"--independent-{phase}-prior-epochs must be nonnegative")
        independent_phase_epochs[phase] = (
            args.independent_prior_epochs
            if configured_epochs is None
            else configured_epochs
        )
    if args.independent_prior_temperature < 0.0:
        parser.error("--independent-prior-temperature must be nonnegative")
    if (
        args.independent_prior_cost_regression
        and args.independent_prior_temperature > 0.0
    ):
        parser.error(
            "--independent-prior-cost-regression cannot be combined with "
            "--independent-prior-temperature"
        )
    if args.independent_crossfit_confidence_z < 0.0:
        parser.error("--independent-crossfit-confidence-z must be nonnegative")
    if args.independent_crossfit_trees < 1:
        parser.error("--independent-crossfit-trees must be positive")
    if args.residual_split_prior_epochs < 0:
        parser.error("--residual-split-prior-epochs must be nonnegative")
    if args.residual_split_prior_epochs > 0 and independent_phase_epochs["dec"] <= 0:
        parser.error("--residual-split-prior-epochs requires Dec independent priors")
    if (
        args.residual_split_prior_epochs > 0
        and args.residual_split_root_objective == "cost_regression"
        and not args.independent_prior_cost_regression
    ):
        parser.error(
            "cost-regression residual split priors require "
            "--independent-prior-cost-regression"
        )
    independent_prior_enabled = args.independent_action_summary is not None and any(
        int(value) > 0 for value in independent_phase_epochs.values()
    )
    if args.independent_action_summary is not None:
        args.independent_action_summary = args.independent_action_summary.resolve()
        if not args.independent_action_summary.is_file():
            parser.error(
                "independent-action summary does not exist: "
                f"{args.independent_action_summary}"
            )
    if (
        any(int(value) > 0 for value in independent_phase_epochs.values())
        and not independent_prior_enabled
    ):
        parser.error("--independent-prior-epochs requires --independent-action-summary")
    if runtime_replay_enabled and not args.replay_query_id:
        parser.error(
            "runtime replay requires at least one --replay-query-id from the "
            "current fold's training set"
        )
    if args.replay_batch_size < 1:
        parser.error("--replay-batch-size must be positive")
    if args.replay_minimum_samples < 1:
        parser.error("--replay-minimum-samples must be at least 1")
    if args.replay_importance_power < 0.0:
        parser.error("--replay-importance-power must be nonnegative")
    if args.sched_replay_temperature <= 0.0:
        parser.error("--sched-replay-temperature must be positive")
    if args.replay_action_cost_temperature < 0.0:
        parser.error("--replay-action-cost-temperature must be nonnegative")
    if args.replay_action_cost_regression and args.replay_action_cost_temperature > 0.0:
        parser.error(
            "--replay-action-cost-regression cannot be combined with "
            "--replay-action-cost-temperature"
        )
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = (
        args.output if args.resume and args.output.is_file() else args.base_checkpoint
    )
    model, prior_metadata, optimizer_state = load_model_checkpoint(
        checkpoint_path,
        hidden=args.hidden,
        device=device,
    )
    initialized_safely = checkpoint_path is None and not args.random_initial_policy
    if initialized_safely:
        initialize_policy_profile(
            model,
            profile=args.initial_policy_profile,
            sched_alpha=args.initial_sched_alpha,
            action_bias=args.initial_action_bias,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_state_restored = False
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_state_restored = True
        except ValueError:
            # Architecture migrations keep model weights but start fresh
            # optimizer moments for newly introduced parameters.
            optimizer_state = None
    prior_iteration = int(prior_metadata.get("iteration") or 0)
    next_iteration = prior_iteration + (0 if args.initialize_only else 1)
    policy_version = args.policy_version or f"online-iter-{next_iteration}"

    losses = {phase: 0.0 for phase in PHASE_LEVEL}
    ppo_diagnostics: dict[str, dict[str, Any]] = {phase: {} for phase in PHASE_LEVEL}
    counts = {phase: 0 for phase in PHASE_LEVEL}
    replay_losses = {phase: 0.0 for phase in PHASE_LEVEL}
    replay_counts = {phase: 0 for phase in PHASE_LEVEL}
    replay_action_counts: dict[str, dict[str, int]] = {
        phase: {} for phase in PHASE_LEVEL
    }
    independent_prior_losses = {phase: 0.0 for phase in PHASE_LEVEL}
    independent_prior_counts = {phase: 0 for phase in PHASE_LEVEL}
    independent_prior_action_counts: dict[str, dict[str, int]] = {
        phase: {} for phase in PHASE_LEVEL
    }
    residual_split_prior_count = 0
    residual_split_prior_losses = {
        "root_cost_regression": 0.0,
        "residual_split": 0.0,
    }
    with ExperienceStore(args.experience_db, read_only=True) as store:
        cutoff_ms = store.now_ms()
        if not args.initialize_only:
            if not args.replay_only:
                buffers = collect_transitions(
                    store,
                    workload=args.workload,
                    policy_version=args.source_policy_version,
                    query_ids=args.replay_query_id,
                    cutoff_ms=cutoff_ms,
                    require_stochastic=True,
                    reward_scale_ms=args.reward_scale_ms,
                    reward_clip=args.reward_clip,
                    action_ablation=args.action_ablation,
                    state_ablation=args.state_ablation,
                    catalog=catalog,
                )
                for phase, transitions in buffers.items():
                    counts[phase] = len(transitions)
                    losses[phase] = ppo_update(
                        model,
                        optimizer,
                        transitions,
                        PHASE_LEVEL[phase],
                        device,
                        method="standardmdp_rl",
                        n_epochs=args.epochs,
                        batch_size=args.batch_size,
                        ent_coef=args.entropy,
                        gamma=args.gamma,
                        lam=args.lambda_gae,
                        freeze_encoder=(phase in FROZEN_REPLAY_ENCODER_PHASES),
                        diagnostics=ppo_diagnostics[phase],
                    )
            if args.replay_only and not (
                runtime_replay_enabled or independent_prior_enabled
            ):
                raise RuntimeError(
                    "--replay-only requires runtime replay or independent priors"
                )
            if runtime_replay_enabled:
                replay_targets = collect_replay_targets(
                    store,
                    workload=args.workload,
                    cutoff_ms=cutoff_ms,
                    reward_scale_ms=args.reward_scale_ms,
                    query_ids=args.replay_query_id,
                    minimum_samples=args.replay_minimum_samples,
                    action_ablation=args.action_ablation,
                    state_ablation=args.state_ablation,
                    catalog=catalog,
                )
                for phase, targets in replay_targets.items():
                    phase_epochs = int(phase_replay_epochs[phase])
                    replay_counts[phase] = len(targets)
                    action_counts: dict[str, int] = defaultdict(int)
                    for target in targets:
                        action_counts[str(target.target)] += 1
                    replay_action_counts[phase] = dict(sorted(action_counts.items()))
                    print(
                        "runtime replay "
                        f"phase={phase} targets={len(targets)} "
                        f"epochs={phase_epochs}",
                        flush=True,
                    )
                    replay_losses[phase] = replay_policy_update(
                        model,
                        optimizer,
                        targets,
                        PHASE_LEVEL[phase],
                        device,
                        epochs=phase_epochs,
                        batch_size=args.replay_batch_size,
                        importance_power=phase_replay_importance_power[phase],
                        freeze_encoder=(phase in FROZEN_REPLAY_ENCODER_PHASES),
                        sched_cost_temperature=(args.sched_replay_temperature),
                        action_cost_temperature=(
                            args.replay_action_cost_temperature or None
                        ),
                        action_cost_regression=(
                            args.replay_action_cost_regression and phase != "sched"
                        ),
                        balance_classes=(
                            False
                            if (
                                args.replay_action_cost_temperature > 0.0
                                or (
                                    args.replay_action_cost_regression
                                    and phase != "sched"
                                )
                            )
                            else None
                        ),
                    )
                    print(
                        "runtime replay "
                        f"phase={phase} loss={replay_losses[phase]:.6f}",
                        flush=True,
                    )
            if independent_prior_enabled:
                independent_targets = collect_independent_action_targets(
                    store,
                    workload=args.workload,
                    summary_path=args.independent_action_summary,
                    reward_scale_ms=args.reward_scale_ms,
                    query_ids=args.replay_query_id,
                    action_ablation=args.action_ablation,
                    prior_mode=args.independent_prior_mode,
                    crossfit_confidence_z=(args.independent_crossfit_confidence_z),
                    crossfit_trees=args.independent_crossfit_trees,
                    seed=args.seed,
                    state_ablation=args.state_ablation,
                    catalog=catalog,
                )
                for phase, targets in independent_targets.items():
                    phase_prior_epochs = int(independent_phase_epochs[phase])
                    independent_prior_counts[phase] = len(targets)
                    action_counts: dict[str, int] = defaultdict(int)
                    for target in targets:
                        action_counts[str(target.target)] += 1
                    independent_prior_action_counts[phase] = dict(
                        sorted(action_counts.items())
                    )
                    print(
                        "independent prior "
                        f"phase={phase} targets={len(targets)} "
                        f"epochs={phase_prior_epochs}",
                        flush=True,
                    )
                    independent_prior_losses[phase] = replay_policy_update(
                        model,
                        optimizer,
                        targets,
                        PHASE_LEVEL[phase],
                        device,
                        epochs=phase_prior_epochs,
                        batch_size=args.replay_batch_size,
                        importance_power=phase_replay_importance_power[phase],
                        freeze_encoder=(
                            phase in FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES
                        ),
                        sched_cost_temperature=(args.sched_replay_temperature),
                        action_cost_temperature=(
                            args.independent_prior_temperature or None
                        ),
                        action_cost_regression=(
                            args.independent_prior_cost_regression and phase != "sched"
                        ),
                        balance_classes=False,
                    )
                if args.residual_split_prior_epochs > 0:
                    residual_targets = collect_residual_split_prior_targets(
                        store,
                        workload=args.workload,
                        query_ids=args.replay_query_id,
                        action_ablation=args.action_ablation,
                        state_ablation=args.state_ablation,
                        catalog=catalog,
                    )
                    residual_split_prior_count = len(residual_targets)
                    if not residual_targets:
                        raise RuntimeError(
                            "residual split prior found no eligible train-only "
                            "Dec states"
                        )
                    residual_split_prior_losses = joint_dec_prior_update(
                        model,
                        optimizer,
                        independent_targets["dec"],
                        residual_targets,
                        device,
                        epochs=args.residual_split_prior_epochs,
                        root_cost_regression=(
                            args.residual_split_root_objective == "cost_regression"
                        ),
                    )
                    print(
                        "residual split prior "
                        f"targets={residual_split_prior_count} "
                        f"epochs={args.residual_split_prior_epochs} "
                        f"losses={residual_split_prior_losses}",
                        flush=True,
                    )
            replay_updated = any(
                replay_counts[phase] > 0 and phase_replay_epochs[phase] > 0
                for phase in PHASE_LEVEL
            )
            independent_prior_updated = any(
                independent_prior_counts[phase] > 0
                and independent_phase_epochs[phase] > 0
                for phase in PHASE_LEVEL
            )
            if (
                not any(counts.values())
                and not replay_updated
                and not independent_prior_updated
            ):
                raise RuntimeError(
                    "no eligible on-policy decisions or deduplicated runtime "
                    "replay targets matched the current training-query whitelist"
                )

        metadata = {
            **prior_metadata,
            "architecture_version": "online-four-head-v10-paper-actions",
            "iteration": next_iteration,
            "policy_version": policy_version,
            "trained_heads": ["dec", "sched", "enum", "adapt"],
            "workload": args.workload,
            "hidden": args.hidden,
            "adapt_labels": ADAPT_LABELS,
            "enum_labels": ENUM_LABELS,
            "sched_alphas": SCHED_ALPHA_VALUES,
            "action_ablation": args.action_ablation,
            "state_ablation": args.state_ablation,
            "experience_db": str(args.experience_db.resolve()),
            "experience_cutoff_ms": cutoff_ms,
            "reward": {
                "objective": "sum_runtime",
                "scale_ms": args.reward_scale_ms,
                "clip": args.reward_clip,
            },
            "transition_counts": counts,
            "losses": losses,
            "ppo_diagnostics": ppo_diagnostics,
            "runtime_replay": {
                "enabled": runtime_replay_enabled,
                "replay_only": args.replay_only,
                "training_query_ids": sorted(set(args.replay_query_id or [])),
                "replay_groups": sorted(set(args.replay_group or [])),
                "split_protocol": args.split_protocol,
                "fold": args.fold,
                "epochs": args.replay_epochs,
                "phase_epochs": phase_replay_epochs,
                "batch_size": args.replay_batch_size,
                "minimum_samples": args.replay_minimum_samples,
                "importance_power": args.replay_importance_power,
                "phase_importance_power": (phase_replay_importance_power),
                "sched_cost_temperature": (args.sched_replay_temperature),
                "action_cost_temperature": (args.replay_action_cost_temperature),
                "action_cost_regression": args.replay_action_cost_regression,
                "target_selection": "minimum_fixed_runtime",
                "target_counts": replay_counts,
                "target_action_counts": replay_action_counts,
                "losses": replay_losses,
                "encoder_frozen_heads": sorted(FROZEN_REPLAY_ENCODER_PHASES),
            },
            "independent_action_prior": {
                "enabled": independent_prior_enabled,
                "summary": (
                    str(args.independent_action_summary)
                    if args.independent_action_summary is not None
                    else None
                ),
                "epochs": args.independent_prior_epochs,
                "phase_epochs": independent_phase_epochs,
                "relative_regret_temperature": (args.independent_prior_temperature),
                "cost_regression": args.independent_prior_cost_regression,
                "mode": args.independent_prior_mode,
                "crossfit_confidence_z": args.independent_crossfit_confidence_z,
                "crossfit_trees": args.independent_crossfit_trees,
                "training_query_ids": sorted(set(args.replay_query_id or [])),
                "target_counts": independent_prior_counts,
                "target_action_counts": independent_prior_action_counts,
                "losses": independent_prior_losses,
                "encoder_frozen_heads": sorted(FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES),
                "state_source": "measured_root_online_states",
                "runtime_source": "fixed_independent_action_summary",
                "residual_split_prior": {
                    "epochs": args.residual_split_prior_epochs,
                    "target_count": residual_split_prior_count,
                    "losses": residual_split_prior_losses,
                    "state_source": "train_only_observed_residual_high_states",
                    "target_action": "split",
                    "joint_root_loss": (
                        "independent_action_cost_regression"
                        if args.residual_split_root_objective == "cost_regression"
                        else "conservative_independent_action_label"
                    ),
                    "root_objective": args.residual_split_root_objective,
                },
            },
            "initial_policy": (
                {
                    "kind": (
                        "postgres_safe"
                        if args.initial_policy_profile == "postgres"
                        else "independent_action_profile"
                    ),
                    "profile": args.initial_policy_profile,
                    "sched_alpha": args.initial_sched_alpha,
                    "action_bias": args.initial_action_bias,
                }
                if initialized_safely
                else prior_metadata.get("initial_policy", {"kind": "random"})
            ),
            "optimizer_state_restored": optimizer_state_restored,
        }
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
            "rng_state": {
                "python": random.getstate(),
                "numpy": portable_numpy_rng_state(),
                "torch": torch.get_rng_state(),
            },
        }
        save_checkpoint_atomic(args.output.resolve(), payload)
        checkpoint_id = content_hash(
            {
                "iteration": next_iteration,
                "policy_version": policy_version,
                "path": str(args.output.resolve()),
                "experience_cutoff_ms": cutoff_ms,
            }
        )

    print(
        json.dumps(
            {
                "checkpoint": str(args.output.resolve()),
                "checkpoint_id": checkpoint_id,
                "iteration": next_iteration,
                "policy_version": policy_version,
                "transitions": counts,
                "losses": losses,
                "ppo_diagnostics": ppo_diagnostics,
                "replay_targets": replay_counts,
                "replay_losses": replay_losses,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
