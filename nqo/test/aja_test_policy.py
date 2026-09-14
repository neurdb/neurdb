"""Deterministic policy used by the executor-level AJA smoke test."""

import json


def predict(state):
    request_type = str(state.get("request_type") or "").lower()
    if request_type == "high":
        return {"high_action": "stop"}
    if request_type == "search":
        return {"search_label": "default"}
    if request_type == "low":
        plan_state = json.dumps(state.get("plan_json") or {}, sort_keys=True)
        return {
            "aja_level": (
                "aggressive"
                if "build_aggressive_case" in plan_state
                else "conservative"
            ),
            "lip_action": "none",
        }
    return {}
