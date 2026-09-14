#!/usr/bin/env python3
"""Current action-space and PPO utilities shared by online NQO."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from optimization.action_vocabulary import (
    LEGACY_ACTION_ABLATIONS,
    canonical_phase,
)

from ..encoders.state import StructuredState

ADAPT_LABELS = [
    "none",
    "filter_selective",
    "ajoin_conservative",
    "filter_selective+ajoin_conservative",
]
ENUM_LABELS = ["native", "top5"]
SCHED_ALPHA_VALUES = (0.0, 0.5, 1.0)
N_DEC = 2
N_SCHED = len(SCHED_ALPHA_VALUES)
N_ENUM = len(ENUM_LABELS)
N_ADAPT = len(ADAPT_LABELS)
ACTION_ABLATIONS = ("none", "no_dec", "no_enum", "no_filter", "no_ajoin")
COVERAGE_EPHEMERAL_STATE_FIELDS = {
    "pid",
    "run_id",
    "relid",
    "cumulative_cost_ms",
    "plan_state_ms",
    "remaining_splits",
}
COVERAGE_TEMP_RELATION_PATTERN = re.compile(r"\btemp[0-9]+\b", re.IGNORECASE)


def coverage_stable_state(
    value: Any,
    temp_relations: Optional[dict[str, str]] = None,
) -> Any:
    """Return the semantic state identity used for coverage exploration."""
    if temp_relations is None:
        temp_relations = {}
    if isinstance(value, dict):
        return {
            key: coverage_stable_state(item, temp_relations)
            for key, item in value.items()
            if key not in COVERAGE_EPHEMERAL_STATE_FIELDS
        }
    if isinstance(value, list):
        return [coverage_stable_state(item, temp_relations) for item in value]
    if isinstance(value, str):

        def canonical_temp(match: re.Match[str]) -> str:
            key = match.group(0).lower()
            if key not in temp_relations:
                temp_relations[key] = f"__nqo_temp_{len(temp_relations) + 1}"
            return temp_relations[key]

        return COVERAGE_TEMP_RELATION_PATTERN.sub(canonical_temp, value)
    return value


def coverage_state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(
        coverage_stable_state(state),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


LEGACY_HEAD_ROW_MAPPINGS = {
    "sched": {5: (0, 2, 4)},
    "enum": {4: (0, 2)},
    "adapt": {6: (0, 2, 3, 5), 9: (0, 2, 6, 8)},
}

LEGACY_CHECKPOINT_KEY_PARTS = {
    "high_actor": "dec_actor",
    "high_critic": "dec_critic",
    "schedule_actor": "sched_actor",
    "schedule_critic": "sched_critic",
    "schedule_adapter": "sched_adapter",
    "search_actor": "enum_actor",
    "search_critic": "enum_critic",
    "low_actor": "adapt_actor",
    "low_critic": "adapt_critic",
    "low_cost_head": "adapt_cost_head",
    "encoder.high_trunk": "encoder.dec_trunk",
    "encoder.search_trunk": "encoder.enum_trunk",
    "encoder.low_trunk": "encoder.adapt_trunk",
}


def migrate_action_space_checkpoint_tensors(
    model: nn.Module,
    state_dict: dict[str, Any],
) -> Tuple[dict[str, Any], List[str]]:
    """Remap older action-head tensors into the current four-head model."""
    renamed_state = {}
    renamed_keys = []
    for original_key, value in state_dict.items():
        key = original_key
        for legacy, canonical in LEGACY_CHECKPOINT_KEY_PARTS.items():
            key = key.replace(legacy, canonical)
        renamed_state[key] = value
        if key != original_key:
            renamed_keys.append(f"{original_key}->{key}")
    state_dict = renamed_state
    target_state = model.state_dict()
    migrated = renamed_keys
    for key, source in list(state_dict.items()):
        target = target_state.get(key)
        if target is None or not hasattr(source, "shape"):
            continue

        head_kind = None
        if "sched_actor" in key:
            head_kind = "sched"
        elif "enum_actor" in key:
            head_kind = "enum"
        elif "adapt_actor" in key or "adapt_cost_head" in key:
            head_kind = "adapt"

        if (
            head_kind is not None
            and len(source.shape) == len(target.shape)
            and source.shape[1:] == target.shape[1:]
        ):
            source_rows = LEGACY_HEAD_ROW_MAPPINGS[head_kind].get(source.shape[0])
            if source_rows is not None and target.shape[0] == len(source_rows):
                remapped = target.clone()
                for target_row, source_row in enumerate(source_rows):
                    remapped[target_row].copy_(source[source_row])
                state_dict[key] = remapped
                migrated.append(key)
                continue

        if (
            key == "encoder.adapt_trunk.0.weight"
            and len(source.shape) == 2
            and source.shape[0] == target.shape[0]
            and source.shape[1] < target.shape[1]
        ):
            expanded = target.clone()
            expanded[:, : source.shape[1]].copy_(source)
            expanded[:, source.shape[1] :].zero_()
            state_dict[key] = expanded
            migrated.append(key)
    return state_dict, migrated


def apply_action_ablation_mask(
    mask: np.ndarray | List[float] | Tuple[float, ...],
    phase: str,
    action_ablation: str = "none",
) -> np.ndarray:
    """Disable one optimizer mechanism while preserving each head's shape."""
    action_ablation = LEGACY_ACTION_ABLATIONS.get(action_ablation, action_ablation)
    phase = canonical_phase(phase)
    if action_ablation not in ACTION_ABLATIONS:
        raise ValueError(
            f"unknown action ablation {action_ablation!r}; "
            f"expected one of {ACTION_ABLATIONS}"
        )
    result = np.asarray(mask, dtype=np.float32).copy()
    if action_ablation == "no_dec" and phase == "dec":
        result[1:] = 0.0
    elif action_ablation == "no_enum" and phase == "enum":
        for index, label in enumerate(ENUM_LABELS):
            if label != "native":
                result[index] = 0.0
    elif action_ablation == "no_filter" and phase == "adapt":
        for index, label in enumerate(ADAPT_LABELS):
            if label.startswith("filter_"):
                result[index] = 0.0
    elif action_ablation == "no_ajoin" and phase == "adapt":
        for index, label in enumerate(ADAPT_LABELS):
            if "ajoin" in label:
                result[index] = 0.0
    if not np.any(result > 0.0):
        raise ValueError(
            f"action ablation {action_ablation!r} leaves no valid {phase} action"
        )
    return result


@dataclass
class Transition:
    """One on-policy decision used by the online PPO update."""

    state: StructuredState
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    mask: np.ndarray
    duration: int = 1
    temperature: float = 1.0
    exploration_epsilon: float = 0.0
    coverage_mix: float = 0.0
    coverage_probs: Optional[np.ndarray] = None


def _base_query_family(qid: str, workload: str = "job") -> str:
    if "_" in qid:
        return qid.split("_", 1)[0]
    match = re.match(r"^([A-Za-z]*\d+)", qid)
    return match.group(1) if match else qid


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    durations: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute duration-aware generalized advantage estimates."""
    count = len(rewards)
    advantages = np.zeros(count, dtype=np.float32)
    returns = np.zeros(count, dtype=np.float32)
    running_advantage = 0.0
    if durations is None:
        durations = np.ones(count, dtype=np.float32)
    else:
        durations = np.asarray(durations, dtype=np.float32)
    for index in reversed(range(count)):
        next_value = 0.0 if index == count - 1 or dones[index] else values[index + 1]
        duration_discount = float(gamma) ** float(max(durations[index], 1.0))
        delta = (
            rewards[index]
            + duration_discount * next_value * (1 - float(dones[index]))
            - values[index]
        )
        running_advantage = (
            delta
            + duration_discount * lam * (1 - float(dones[index])) * running_advantage
        )
        advantages[index] = running_advantage
        returns[index] = running_advantage + values[index]
    return advantages, returns
