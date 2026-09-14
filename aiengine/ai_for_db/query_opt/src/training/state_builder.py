"""Construction of the structured model states stored in execution experience."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from model.encoders.query_graph import CatalogInfo
from model.encoders.state import (
    DEC_CTX_DIM,
    StructuredState,
    build_adapt_context,
    build_dec_context,
    build_transfer_graph_state,
    empty_plan_tree,
    empty_query_graph_state,
    empty_structured_state,
    flatten_plan_tree_topology,
    normalize_db_plan_json,
    parse_query_graph,
    plan_to_tree,
    remove_query_graph_topology,
)
from optimization.action_vocabulary import canonical_phase, normalize_policy_state
from optimization.state import runtime_relation_plan

STATE_ABLATIONS = (
    "none",
    "no_query_topology",
    "no_plan_topology",
)


class ExecutionStateBuilder:
    """Build exactly the structured states used by online inference."""

    def __init__(
        self,
        workload: str,
        state_ablation: str = "none",
        *,
        catalog: CatalogInfo | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        if state_ablation not in STATE_ABLATIONS:
            raise ValueError(
                f"unknown state ablation {state_ablation!r}; "
                f"expected one of {STATE_ABLATIONS}"
            )
        if catalog is not None and catalog_path is not None:
            raise ValueError("pass catalog or catalog_path, not both")
        if catalog is None:
            selected_path = catalog_path or os.environ.get("NQO_CATALOG_PATH")
            if selected_path is None:
                raise ValueError(
                    "ExecutionStateBuilder requires a database-derived catalog "
                    "snapshot via catalog, catalog_path, or NQO_CATALOG_PATH"
                )
            catalog = CatalogInfo(selected_path)
        self.catalog = catalog
        self.state_ablation = state_ablation
        self.cache: dict[tuple[str, str], StructuredState] = {}

    def build(
        self, phase: str, state_blob_hash: str, state: dict[str, Any]
    ) -> StructuredState:
        phase = canonical_phase(phase)
        state = normalize_policy_state(state)
        # Labels intentionally use a stable semantic state hash, while the
        # model input still includes dynamic context such as cumulative time.
        # Cache by the raw state blob so those two identities do not collapse.
        key = (phase, state_blob_hash)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if phase == "adapt":
            plan_json = normalize_db_plan_json(
                state.get("plan_json") or state.get("plan")
            )
            ctx = build_adapt_context(
                cumulative_ms=float(state.get("cumulative_cost_ms") or 0.0),
                round_index=float(state.get("round") or 0.0),
                max_rounds=float(state.get("max_split_rounds") or 1.0),
                is_split_execution=bool(state.get("is_split_execution", False)),
                enum_action=str(state.get("enum_action") or "native"),
                enum_k=int(state.get("enum_k") or 0),
            )
            plan_tree = plan_to_tree(plan_json, catalog=self.catalog)
            if self.state_ablation == "no_plan_topology":
                plan_tree = flatten_plan_tree_topology(plan_tree)
            structured = StructuredState(
                level="adapt",
                query_graph=empty_query_graph_state(),
                current_plan=plan_tree,
                ctx=ctx,
                cache_key=(
                    "online",
                    phase,
                    self.state_ablation,
                    state_blob_hash,
                    ctx.tobytes(),
                ),
            )
            self.cache[key] = structured
            return structured

        level = "dec" if phase in {"dec", "sched"} else "enum"
        sql = str(state.get("sql") or state.get("original_sql") or "")
        plan_json = normalize_db_plan_json(
            state.get("plan_json") or state.get("plan") or runtime_relation_plan(state)
        )
        try:
            graph = parse_query_graph(sql)
            query_graph, _stats = build_transfer_graph_state(
                sql,
                graph,
                self.catalog,
                plan_json=(None if level == "dec" else plan_json),
            )
            if self.state_ablation == "no_query_topology":
                query_graph = remove_query_graph_topology(query_graph)
        except Exception:
            ctx_dim = DEC_CTX_DIM if level == "dec" else 0
            structured = empty_structured_state(level=level, ctx_dim=ctx_dim)
            self.cache[key] = structured
            return structured

        ctx = np.zeros(DEC_CTX_DIM if level == "dec" else 0, dtype=np.float32)
        if level == "dec":
            ctx = build_dec_context(
                cumulative_ms=float(state.get("cumulative_cost_ms") or 0.0),
                round_index=float(state.get("round") or 0.0),
                max_rounds=float(state.get("max_split_rounds") or 1.0),
            )
        structured = StructuredState(
            level=level,
            query_graph=query_graph,
            current_plan=empty_plan_tree(),
            ctx=ctx,
            cache_key=("online", phase, self.state_ablation, state_blob_hash),
        )
        self.cache[key] = structured
        return structured
