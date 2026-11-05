"""Shared transformations for DB-supplied optimization state."""

from __future__ import annotations

from typing import Any, Optional


def runtime_relation_plan(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Represent live relation estimates as a plan-shaped feature input."""
    children = []
    for relation in state.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        children.append(
            {
                "Node Type": "Seq Scan",
                "Relation Name": relation.get("relname"),
                "Alias": relation.get("alias"),
                "Plan Rows": max(float(relation.get("estimated_rows") or 0.0), 0.0),
                "Plan Width": 0,
                "Startup Cost": 0.0,
                "Total Cost": float(relation.get("pages") or 0.0),
            }
        )
    if not children:
        return None
    return {
        "Plan": {
            "Node Type": "Append",
            "Plan Rows": sum(item["Plan Rows"] for item in children),
            "Plan Width": 0,
            "Startup Cost": 0.0,
            "Total Cost": sum(item["Total Cost"] for item in children),
            "Plans": children,
        }
    }
