"""Canonical NQO action vocabulary and legacy protocol adapters.

The public names in this module follow the paper: Dec, Sched, Enum, Filter,
and AJoin.  Older checkpoints and execution buffers used the implementation
labels high, select, search, low, LIP, and AJA; readers normalize those labels
at the boundary so released artifacts remain usable.
"""

from __future__ import annotations

from typing import Any, Mapping

DEC_PHASE = "dec"
SCHED_PHASE = "sched"
ENUM_PHASE = "enum"
ADAPT_PHASE = "adapt"
DECISION_PHASES = (DEC_PHASE, SCHED_PHASE, ENUM_PHASE, ADAPT_PHASE)

LEGACY_PHASE_ALIASES = {
    "high": DEC_PHASE,
    "select": SCHED_PHASE,
    "schedule": SCHED_PHASE,
    "search": ENUM_PHASE,
    "low": ADAPT_PHASE,
}

LEGACY_ACTION_ABLATIONS = {"no_split": "no_dec", "no_topk": "no_enum"}


def canonical_phase(value: Any) -> str:
    """Return the paper-aligned phase name for new or legacy input."""
    phase = str(value or "").strip().lower()
    return LEGACY_PHASE_ALIASES.get(phase, phase)


def canonical_dec_action(value: Any) -> str:
    """Normalize Dec to ``apply`` or ``skip``."""
    label = str(value or "skip").strip().lower()
    if label in {"apply", "split", "on", "true", "1"}:
        return "apply"
    if label in {"skip", "stop", "none", "off", "false", "0"}:
        return "skip"
    raise ValueError(f"unsupported Dec action: {value!r}")


def canonical_enum_action(value: Any, k: Any = None) -> str:
    """Normalize Enum to ``native`` or a ``topN`` label."""
    label = str(value or "native").strip().lower().replace("-", "")
    if label in {"default", "none", "native", "off"}:
        return "native"
    if label in {"split", "top1"}:
        return "top1"
    if label == "topk":
        return f"top{int(k or 5)}"
    if label.startswith("top") and label[3:].isdigit():
        return label
    if label == "left_deep":
        return label
    raise ValueError(f"unsupported Enum action: {value!r}")


def enum_action_to_strategy(value: Any, k: Any = None) -> tuple[str, int]:
    """Translate a canonical Enum action into PostgreSQL planner controls."""
    action = canonical_enum_action(value, k)
    if action == "native":
        return "default", 1
    if action.startswith("top") and action[3:].isdigit():
        return "topk", int(action[3:])
    return action, int(k or 1)


def canonical_filter_action(value: Any) -> str:
    """Normalize Filter to ``none``, ``selective``, or ``full``."""
    label = str(value or "none").strip().lower()
    aliases = {
        "off": "none",
        "lip_sel": "selective",
        "lip_selective": "selective",
        "lip_full": "full",
    }
    label = aliases.get(label, label)
    if label not in {"none", "selective", "full"}:
        raise ValueError(f"unsupported Filter action: {value!r}")
    return label


def canonical_ajoin_action(value: Any) -> str:
    """Normalize AJoin to ``off``, ``conservative``, or ``aggressive``."""
    label = str(value or "off").strip().lower()
    aliases = {"none": "off", "aja": "aggressive"}
    label = aliases.get(label, label)
    if label not in {"off", "conservative", "aggressive"}:
        # Static join-method controls predate the learned AJoin action.  Keep
        # accepting them for old fixed-policy traces.
        if label not in {"hashjoin", "nestloop", "mergejoin"}:
            raise ValueError(f"unsupported AJoin action: {value!r}")
    return label


def adapt_label(filter_action: Any, ajoin_action: Any) -> str:
    """Return the canonical joint Filter/AJoin label."""
    filter_label = canonical_filter_action(filter_action)
    ajoin_label = canonical_ajoin_action(ajoin_action)
    parts = []
    if filter_label != "none":
        parts.append(f"filter_{filter_label}")
    if ajoin_label != "off":
        parts.append(f"ajoin_{ajoin_label}")
    return "+".join(parts) or "none"


def split_adapt_label(value: Any) -> tuple[str, str]:
    """Translate canonical or legacy joint labels into Filter and AJoin."""
    label = str(value or "none").strip().lower()
    legacy = {
        "lip_sel": "filter_selective",
        "lip_full": "filter_full",
        "aja": "ajoin_aggressive",
        "aja_conservative": "ajoin_conservative",
        "lip_sel+aja": "filter_selective+ajoin_aggressive",
        "lip_full+aja": "filter_full+ajoin_aggressive",
        "lip_sel+aja_conservative": "filter_selective+ajoin_conservative",
        "lip_full+aja_conservative": "filter_full+ajoin_conservative",
    }
    label = legacy.get(label, label)
    filter_action = "none"
    ajoin_action = "off"
    for component in label.split("+"):
        if component.startswith("filter_"):
            filter_action = canonical_filter_action(component.removeprefix("filter_"))
        elif component.startswith("ajoin_"):
            ajoin_action = canonical_ajoin_action(component.removeprefix("ajoin_"))
        elif component not in {"", "none"}:
            raise ValueError(f"unsupported adaptation action: {value!r}")
    return filter_action, ajoin_action


def normalize_policy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical state without mutating stored legacy input."""
    normalized = dict(state)
    normalized["request_type"] = canonical_phase(normalized.get("request_type"))
    if any(
        key in normalized for key in ("enum_action", "search_label", "search_strategy")
    ):
        normalized["enum_action"] = canonical_enum_action(
            normalized.get("enum_action")
            or normalized.get("search_label")
            or normalized.get("search_strategy"),
            normalized.get("enum_k", normalized.get("search_k")),
        )
        enum_action = normalized["enum_action"]
        normalized["enum_k"] = int(
            normalized.get(
                "enum_k",
                normalized.get(
                    "search_k",
                    enum_action[3:] if enum_action.startswith("top") else 1,
                ),
            )
        )
    for legacy_key in ("search_label", "search_strategy", "search_k"):
        normalized.pop(legacy_key, None)
    return normalized


def normalize_policy_action(
    action: Mapping[str, Any],
    *,
    phase: Any = None,
) -> dict[str, Any]:
    """Return canonical action fields for new or released legacy records."""
    normalized = dict(action)
    canonical = canonical_phase(phase)

    if canonical == DEC_PHASE or any(
        key in normalized for key in ("dec_action", "high_action")
    ):
        raw = normalized.get("dec_action", normalized.get("high_action"))
        if raw is None and normalized.get("action") in {
            "apply",
            "skip",
            "split",
            "stop",
        }:
            raw = normalized["action"]
        if raw is not None:
            normalized["dec_action"] = canonical_dec_action(raw)

    if "sched_alpha" not in normalized and "schedule_alpha" in normalized:
        normalized["sched_alpha"] = normalized["schedule_alpha"]
    if "sched_idx" not in normalized and "schedule_idx" in normalized:
        normalized["sched_idx"] = normalized["schedule_idx"]
    if (
        canonical == SCHED_PHASE
        and normalized.get("candidate_id") is not None
        and normalized.get("sched_alpha") is None
    ):
        # Released QuerySplit buffers omitted alpha when the selected
        # candidate was recorded directly; those runs used alpha=0.5.
        normalized["sched_alpha"] = 0.5

    if canonical == ENUM_PHASE or any(
        key in normalized for key in ("enum_action", "search_label", "search_strategy")
    ):
        raw = normalized.get("enum_action")
        if raw is None:
            raw = normalized.get("search_label", normalized.get("search_strategy"))
        if raw is not None:
            normalized["enum_action"] = canonical_enum_action(
                raw,
                normalized.get("enum_k", normalized.get("search_k")),
            )

    if "filter_action" not in normalized and "lip_action" in normalized:
        normalized["filter_action"] = canonical_filter_action(normalized["lip_action"])
    if "ajoin_action" not in normalized:
        raw_ajoin = normalized.get("aja_level", normalized.get("execution_action"))
        if raw_ajoin is not None:
            normalized["ajoin_action"] = canonical_ajoin_action(raw_ajoin)
    if "adapt_action" not in normalized and "low_label" in normalized:
        filter_action, ajoin_action = split_adapt_label(normalized["low_label"])
        normalized.setdefault("filter_action", filter_action)
        normalized.setdefault("ajoin_action", ajoin_action)
    if canonical == ADAPT_PHASE or any(
        key in normalized for key in ("filter_action", "ajoin_action")
    ):
        normalized.setdefault("filter_action", "none")
        normalized.setdefault("ajoin_action", "off")
        normalized["adapt_action"] = adapt_label(
            normalized["filter_action"], normalized["ajoin_action"]
        )
    for legacy_key in (
        "high_action",
        "schedule_alpha",
        "schedule_idx",
        "search_label",
        "search_strategy",
        "search_k",
        "lip_action",
        "aja_level",
        "execution_action",
        "low_label",
    ):
        normalized.pop(legacy_key, None)
    if "enum_action" in normalized:
        enum_action = normalized["enum_action"]
        normalized["enum_k"] = (
            1
            if enum_action == "native"
            else int(
                normalized.get(
                    "enum_k",
                    enum_action[3:] if enum_action.startswith("top") else 1,
                )
            )
        )
    return normalized
