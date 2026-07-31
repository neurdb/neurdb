"""Environment-configurable policy for isolated NeurQO action experiments.

The module is loaded by ai_server.py through:

    --model-module /code/neurdb-dev/neurqo/server/fixed_policy.py:predict

Each primitive can be enabled independently without changing DB code.
"""

from __future__ import annotations

import os
from typing import Any


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


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
    request_type = str(state.get("request_type") or "").lower()

    if request_type == "high":
        high = _setting("NEURQO_FIXED_HIGH", "stop")
        round_no = int(state.get("round") or 0)
        remaining = int(state.get("remaining_splits") or 0)
        max_split_rounds = _integer("NEURQO_FIXED_SPLIT_ROUNDS", -1)
        split = high in {"split", "apply", "on", "true", "1"} and remaining > 0
        if max_split_rounds >= 0 and round_no >= max_split_rounds:
            split = False
        return {
            "high_action": "split" if split else "stop",
            "order_decision": _setting("NEURQO_FIXED_ORDER_DECISION", "only_cost"),
            "note": "fixed high policy",
        }

    if request_type == "select":
        alpha = _float("NEURQO_FIXED_ALPHA", 0.5)
        sequence = _float_sequence("NEURQO_FIXED_ALPHA_SEQUENCE")
        round_no = int(state.get("round") or 0)
        if round_no < len(sequence):
            alpha = sequence[round_no]
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("fixed schedule alpha must be in [0,1]")
        return {
            "schedule_alpha": alpha,
            "selection_strategy": f"alpha_{alpha:.2f}",
            "note": (
                "fixed alpha sequence scheduler"
                if sequence
                else "fixed alpha scheduler"
            ),
        }

    if request_type == "search":
        search = _setting("NEURQO_FIXED_SEARCH", "default")
        if search in {"split", "top1"}:
            search_label = "split"
        elif search in {"top5", "top10", "default"}:
            search_label = search
        elif search == "topk":
            search_label = f"top{_integer('NEURQO_FIXED_SEARCH_K', 5)}"
        else:
            raise ValueError(f"unsupported fixed search action: {search}")
        return {
            "search_label": search_label,
            "note": "fixed search policy",
        }

    if request_type == "low":
        lip = _setting("NEURQO_FIXED_LIP", "none")
        aja = _setting("NEURQO_FIXED_AJA", "none")
        if lip not in {"none", "full", "selective"}:
            raise ValueError(f"unsupported fixed LIP action: {lip}")
        if aja not in {"none", "conservative", "aggressive"}:
            raise ValueError(f"unsupported fixed AJA action: {aja}")
        return {
            "lip_action": lip,
            "aja_level": aja,
            "note": "fixed execution policy",
        }

    return {}
