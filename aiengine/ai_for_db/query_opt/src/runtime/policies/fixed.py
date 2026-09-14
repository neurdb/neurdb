"""Environment-configurable policy for isolated NQO action experiments.

The module is loaded by the action server through:

    --model-module runtime.policies.fixed:predict

Each primitive can be enabled independently without changing DB code.
"""

from __future__ import annotations

import os
from typing import Any

from optimization.action_vocabulary import (
    ADAPT_PHASE,
    DEC_PHASE,
    ENUM_PHASE,
    SCHED_PHASE,
    adapt_label,
    canonical_ajoin_action,
    canonical_dec_action,
    canonical_enum_action,
    canonical_filter_action,
    canonical_phase,
)


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _setting_with_legacy(name: str, legacy: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        value = os.environ.get(legacy, default)
    return value.strip().lower()


def _integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _float_sequence(name: str) -> list[float]:
    value = os.environ.get(name, "")
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def predict(state: dict[str, Any]) -> dict[str, Any]:
    request_type = canonical_phase(state.get("request_type"))

    if request_type == DEC_PHASE:
        dec = canonical_dec_action(
            _setting_with_legacy("NQO_FIXED_DEC", "NQO_FIXED_HIGH", "skip")
        )
        round_no = int(state.get("round") or 0)
        remaining = int(state.get("remaining_splits") or 0)
        max_dec_rounds = _integer("NQO_FIXED_DEC_ROUNDS", -1)
        if "NQO_FIXED_DEC_ROUNDS" not in os.environ:
            max_dec_rounds = _integer("NQO_FIXED_SPLIT_ROUNDS", -1)
        apply = dec == "apply" and remaining > 0
        if max_dec_rounds >= 0 and round_no >= max_dec_rounds:
            apply = False
        dec_action = "apply" if apply else "skip"
        return {
            "dec_action": dec_action,
            "order_decision": _setting("NQO_FIXED_ORDER_DECISION", "only_cost"),
            "note": "fixed Dec policy",
        }

    if request_type == SCHED_PHASE:
        alpha = _float("NQO_FIXED_SCHED_ALPHA", 0.5)
        if "NQO_FIXED_SCHED_ALPHA" not in os.environ:
            alpha = _float("NQO_FIXED_ALPHA", 0.5)
        sequence = _float_sequence("NQO_FIXED_SCHED_ALPHA_SEQUENCE")
        if not sequence:
            sequence = _float_sequence("NQO_FIXED_ALPHA_SEQUENCE")
        round_no = int(state.get("round") or 0)
        if round_no < len(sequence):
            alpha = sequence[round_no]
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("fixed Sched alpha must be in [0,1]")
        return {
            "sched_alpha": alpha,
            "selection_strategy": f"alpha_{alpha:.2f}",
            "note": (
                "fixed alpha sequence scheduler"
                if sequence
                else "fixed alpha scheduler"
            ),
        }

    if request_type == ENUM_PHASE:
        enum = _setting_with_legacy("NQO_FIXED_ENUM", "NQO_FIXED_SEARCH", "native")
        enum_k = _integer("NQO_FIXED_ENUM_K", 5)
        if "NQO_FIXED_ENUM_K" not in os.environ:
            enum_k = _integer("NQO_FIXED_SEARCH_K", 5)
        enum_action = canonical_enum_action(enum, enum_k)
        return {
            "enum_action": enum_action,
            "enum_k": enum_k if enum_action != "native" else 1,
            "note": "fixed Enum policy",
        }

    if request_type == ADAPT_PHASE:
        filter_action = canonical_filter_action(
            _setting_with_legacy("NQO_FIXED_FILTER", "NQO_FIXED_LIP", "none")
        )
        ajoin_action = canonical_ajoin_action(
            _setting_with_legacy("NQO_FIXED_AJOIN", "NQO_FIXED_AJA", "off")
        )
        return {
            "filter_action": filter_action,
            "ajoin_action": ajoin_action,
            "adapt_action": adapt_label(filter_action, ajoin_action),
            "note": "fixed Adapt policy",
        }

    return {}
