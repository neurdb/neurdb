"""Deterministic policy used by the executor-level AJA smoke test."""

import json

from optimization.action_vocabulary import canonical_phase


def predict(state):
    request_type = canonical_phase(state.get("request_type"))
    if request_type == "dec":
        return {"dec_action": "skip"}
    if request_type == "enum":
        return {"enum_action": "native"}
    if request_type == "adapt":
        plan_state = json.dumps(state.get("plan_json") or {}, sort_keys=True)
        return {
            "ajoin_action": (
                "aggressive"
                if "build_aggressive_case" in plan_state
                else "conservative"
            ),
            "filter_action": "none",
        }
    return {}
