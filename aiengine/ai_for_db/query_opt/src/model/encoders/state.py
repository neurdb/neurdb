#!/usr/bin/env python3
"""Structured query-state construction and encoders for hierarchical RL."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_graph import (
    GRAPH_EMB_DIM,
    CatalogInfo,
    JoinGraph,
    parse_subquery_graph,
)

# ── Dimensions / IDs ────────────────────────────────────────────────────────

TABLE_NODE_FEAT_DIM = 5
COLUMN_NODE_FEAT_DIM = 19
JOIN_EDGE_FEAT_DIM = 2
TABLE_JOIN_EDGE_FEAT_DIM = 1
PLAN_NODE_FEAT_DIM = 10
PLAN_EMB_DIM = 64
DEC_CTX_DIM = 2
ADAPT_ENUM_ACTIONS = ("native", "top1", "top5", "top10")
ADAPT_CTX_DIM = DEC_CTX_DIM + 1 + len(ADAPT_ENUM_ACTIONS)

DEC_STATE_DIM = GRAPH_EMB_DIM + DEC_CTX_DIM
ENUM_STATE_DIM = GRAPH_EMB_DIM
ADAPT_STATE_DIM = PLAN_EMB_DIM + ADAPT_CTX_DIM

TYPE_FAMILY_IDS = {
    "numeric": 0,
    "string": 1,
    "date": 2,
    "bool": 3,
    "other": 4,
}
KEY_ROLE_IDS = {
    "none": 0,
    "pk": 1,
    "fk": 2,
    "pk_fk": 3,
}
FILTER_OP_IDS = {
    "none": 0,
    "eq": 1,
    "range": 2,
    "like": 3,
    "in": 4,
    "other": 5,
}
JOIN_OP_IDS = {
    "eq": 0,
    "range": 1,
    "other": 2,
}

PLAN_OP_TYPES = [
    "seq_scan",
    "index_scan",
    "bitmap_scan",
    "hash_join",
    "nested_loop",
    "merge_join",
    "sort",
    "aggregate",
    "hash",
    "materialize",
    "append",
    "limit",
    "other",
]
PLAN_OTHER_IDX = PLAN_OP_TYPES.index("other")

PLAN_NODE_NAME_TO_EXPLAIN = {
    "Agg": "Aggregate",
    "Append": "Append",
    "BitmapHeapScan": "Bitmap Heap Scan",
    "BitmapIndexScan": "Bitmap Index Scan",
    "CteScan": "CTE Scan",
    "FunctionScan": "Function Scan",
    "Gather": "Gather",
    "GatherMerge": "Gather Merge",
    "Group": "Group",
    "Hash": "Hash",
    "HashJoin": "Hash Join",
    "IndexOnlyScan": "Index Only Scan",
    "IndexScan": "Index Scan",
    "Limit": "Limit",
    "Material": "Materialize",
    "MergeAppend": "Merge Append",
    "MergeJoin": "Merge Join",
    "NestLoop": "Nested Loop",
    "Result": "Result",
    "SeqScan": "Seq Scan",
    "Sort": "Sort",
    "SubqueryScan": "Subquery Scan",
    "TidScan": "Tid Scan",
    "ValuesScan": "Values Scan",
}

# column node layout
COLUMN_IDX_TYPE_FAMILY = 0
COLUMN_IDX_NULL_FRAC = 1
COLUMN_IDX_NDV_RATIO = 2
COLUMN_IDX_AVG_WIDTH = 3
COLUMN_IDX_KEY_ROLE = 4
COLUMN_IDX_HAS_INDEX = 5
COLUMN_IDX_NUM_FILTERS = 6
COLUMN_IDX_FILTER_OP = 7
COLUMN_IDX_CONST_POS = 8
COLUMN_IDX_NUM_DIST = slice(9, 14)
COLUMN_IDX_CAT_DIST = slice(14, 19)


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    max_degree: float
    avg_degree: float
    edge_density: float
    is_cyclic: bool
    star_score: float


@dataclass
class PlanTree:
    op_type_id: int
    node_numeric_feat: np.ndarray
    children: Tuple["PlanTree", ...]
    is_sentinel: bool = False
    _tensor_cache: Dict[Tuple[str, str], torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )

    def op_tensor(self, device: torch.device) -> torch.Tensor:
        key = ("op", str(device))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = torch.tensor(
                self.op_type_id, dtype=torch.long, device=device
            )
        return self._tensor_cache[key]

    def feat_tensor(self, device: torch.device) -> torch.Tensor:
        key = ("feat", str(device))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (
                torch.from_numpy(self.node_numeric_feat).float().to(device)
            )
        return self._tensor_cache[key]


@dataclass
class QueryGraphState:
    table_node_features: np.ndarray
    column_node_features: np.ndarray
    membership_edges: np.ndarray
    join_edges: np.ndarray
    join_edge_features: np.ndarray
    table_join_edges: np.ndarray
    table_join_edge_features: np.ndarray
    _tensor_cache: Dict[Tuple[str, str, str], torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )

    def tensor(
        self,
        name: str,
        arr: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = (name, str(device), str(dtype))
        if key not in self._tensor_cache:
            t = torch.from_numpy(arr)
            if dtype in (torch.long, torch.int64):
                t = t.long()
            else:
                t = t.float()
            self._tensor_cache[key] = t.to(device)
        return self._tensor_cache[key]


@dataclass
class StructuredState:
    level: str
    query_graph: QueryGraphState
    current_plan: PlanTree
    ctx: np.ndarray
    cache_key: Any = None
    _tensor_cache: Dict[Tuple[str, str], torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )

    def tensor(self, name: str, arr: np.ndarray, device: torch.device) -> torch.Tensor:
        key = (name, str(device))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = torch.from_numpy(arr).float().to(device)
        return self._tensor_cache[key]


def empty_plan_tree() -> PlanTree:
    return PlanTree(
        op_type_id=PLAN_OTHER_IDX,
        node_numeric_feat=np.zeros(PLAN_NODE_FEAT_DIM, dtype=np.float32),
        children=(),
        is_sentinel=True,
    )


def empty_query_graph_state() -> QueryGraphState:
    return QueryGraphState(
        table_node_features=np.zeros((1, TABLE_NODE_FEAT_DIM), dtype=np.float32),
        column_node_features=np.zeros((1, COLUMN_NODE_FEAT_DIM), dtype=np.float32),
        membership_edges=np.array([[0, 0]], dtype=np.int64),
        join_edges=np.zeros((0, 2), dtype=np.int64),
        join_edge_features=np.zeros((0, JOIN_EDGE_FEAT_DIM), dtype=np.float32),
        table_join_edges=np.zeros((0, 2), dtype=np.int64),
        table_join_edge_features=np.zeros(
            (0, TABLE_JOIN_EDGE_FEAT_DIM), dtype=np.float32
        ),
    )


def remove_query_graph_topology(graph: QueryGraphState) -> QueryGraphState:
    """Keep query-node content while removing the join-graph structure.

    Table/column features and their schema membership remain available.  Only
    column- and table-level join edges are removed, yielding a bag of relation
    features rather than an uninformative all-zero state.
    """
    return QueryGraphState(
        table_node_features=graph.table_node_features,
        column_node_features=graph.column_node_features,
        membership_edges=graph.membership_edges,
        join_edges=np.zeros((0, 2), dtype=np.int64),
        join_edge_features=np.zeros((0, JOIN_EDGE_FEAT_DIM), dtype=np.float32),
        table_join_edges=np.zeros((0, 2), dtype=np.int64),
        table_join_edge_features=np.zeros(
            (0, TABLE_JOIN_EDGE_FEAT_DIM), dtype=np.float32
        ),
    )


def flatten_plan_tree_topology(tree: PlanTree) -> PlanTree:
    """Keep every plan node/feature while removing parent-child hierarchy.

    The original root stays the root and every descendant becomes one direct
    leaf child.  The plan encoder therefore receives the same operator and
    numeric content as a bag, but cannot use the original tree topology.
    """
    if tree is None or tree.is_sentinel:
        return tree

    descendants: List[PlanTree] = []

    def collect(node: PlanTree) -> None:
        for child in node.children:
            descendants.append(
                PlanTree(
                    op_type_id=child.op_type_id,
                    node_numeric_feat=child.node_numeric_feat,
                    children=(),
                    is_sentinel=child.is_sentinel,
                )
            )
            collect(child)

    collect(tree)
    return PlanTree(
        op_type_id=tree.op_type_id,
        node_numeric_feat=tree.node_numeric_feat,
        children=tuple(descendants),
        is_sentinel=tree.is_sentinel,
    )


def empty_structured_state(
    level: str = "dec", ctx_dim: int = DEC_CTX_DIM
) -> StructuredState:
    return StructuredState(
        level=level,
        query_graph=empty_query_graph_state(),
        current_plan=empty_plan_tree(),
        ctx=np.zeros(ctx_dim, dtype=np.float32),
        cache_key=("empty", level, ctx_dim),
    )


def normalize_db_plan_json(plan: Any) -> Any:
    """Convert compact plan JSON into PostgreSQL EXPLAIN JSON."""
    if plan is None or not isinstance(plan, dict):
        return plan
    if "Plan" in plan:
        return {**plan, "Plan": normalize_db_plan_json(plan.get("Plan"))}
    if "Node Type" in plan:
        out = dict(plan)
        if isinstance(out.get("Plans"), list):
            out["Plans"] = [normalize_db_plan_json(child) for child in out["Plans"]]
        return out
    if "node" not in plan:
        return plan

    out = {
        "Node Type": PLAN_NODE_NAME_TO_EXPLAIN.get(
            str(plan.get("node") or "Other"),
            str(plan.get("node") or "Other"),
        ),
        "Plan Rows": float(plan.get("rows") or 0.0),
        "Startup Cost": float(plan.get("startup_cost") or 0.0),
        "Total Cost": float(plan.get("total_cost") or 0.0),
        "Plan Width": int(plan.get("width") or 0),
    }
    if plan.get("alias"):
        out["Alias"] = str(plan["alias"])
    children = plan.get("children") or []
    if isinstance(children, list) and children:
        out["Plans"] = [normalize_db_plan_json(child) for child in children]
    return out


def build_dec_context(
    *,
    cumulative_ms: float,
    round_index: float,
    max_rounds: float,
) -> np.ndarray:
    """Build the plan-independent context used by Dec and Sched."""
    ctx = np.zeros(DEC_CTX_DIM, dtype=np.float32)
    ctx[0] = np.log1p(max(float(cumulative_ms), 0.0)) / 12.0
    ctx[1] = min(
        max(float(round_index), 0.0) / max(float(max_rounds), 1.0),
        1.0,
    )
    return ctx


def build_adapt_context(
    *,
    cumulative_ms: float,
    round_index: float,
    max_rounds: float,
    is_split_execution: bool,
    enum_action: str,
    enum_k: int = 0,
) -> np.ndarray:
    """Build plan-side execution context without adding a query graph."""
    ctx = np.zeros(ADAPT_CTX_DIM, dtype=np.float32)
    ctx[:DEC_CTX_DIM] = build_dec_context(
        cumulative_ms=cumulative_ms,
        round_index=round_index,
        max_rounds=max_rounds,
    )
    ctx[DEC_CTX_DIM] = 1.0 if is_split_execution else 0.0

    strategy = str(enum_action or "native").strip().lower()
    if strategy in {"topk", "default", "split"}:
        if strategy == "default":
            strategy = "native"
        elif strategy == "split":
            strategy = "top1"
        elif int(enum_k or 0) >= 10:
            strategy = "top10"
        elif int(enum_k or 0) >= 5:
            strategy = "top5"
        else:
            strategy = "top1"
    if strategy not in ADAPT_ENUM_ACTIONS:
        strategy = "native"
    strategy_offset = DEC_CTX_DIM + 1
    ctx[strategy_offset + ADAPT_ENUM_ACTIONS.index(strategy)] = 1.0
    return ctx


def _encode_unique_structured_batch(
    states: List[StructuredState],
    device: torch.device,
    encode_fn,
) -> torch.Tensor:
    unique_states: List[StructuredState] = []
    inverse: List[int] = []
    seen: Dict[Any, int] = {}

    for state in states:
        key = getattr(state, "cache_key", None)
        if key is None:
            key = ("obj", id(state))
        pos = seen.get(key)
        if pos is None:
            pos = len(unique_states)
            seen[key] = pos
            unique_states.append(state)
        inverse.append(pos)

    unique_emb = torch.stack(
        [encode_fn(state, device) for state in unique_states], dim=0
    )
    if len(unique_states) == len(states):
        return unique_emb
    inverse_t = torch.tensor(inverse, dtype=torch.long, device=unique_emb.device)
    return unique_emb.index_select(0, inverse_t)


def parse_query_graph(sql: str) -> JoinGraph:
    graph = parse_subquery_graph(sql)
    if graph.n_nodes == 0:
        graph = JoinGraph()
        graph.aliases = {"t0": "qs_final"}
        graph.edges = []
        graph.predicates = {}
    return graph


def _extract_where_text(sql: str) -> str:
    m = re.search(
        r"\bWHERE\b(.*?)(?:GROUP|ORDER|LIMIT|;|$)", sql, re.DOTALL | re.IGNORECASE
    )
    return m.group(1) if m else ""


def _split_and_clauses(text: str) -> List[str]:
    return [
        c.strip() for c in re.split(r"\bAND\b", text, flags=re.IGNORECASE) if c.strip()
    ]


def _extract_alias_column_refs(sql: str, aliases: List[str]) -> Dict[str, List[str]]:
    alias_set = set(a.lower() for a in aliases)
    cols = {a: set() for a in alias_set}
    for m in re.finditer(r"\b(\w+)\.(\w+)\b", sql):
        alias = m.group(1).lower()
        col = m.group(2).lower()
        if alias in alias_set:
            cols[alias].add(col)
    return {a: sorted(v) for a, v in cols.items()}


def _classify_filter_clause(clause: str) -> int:
    cl = f" {clause.lower()} "
    if any(tok in cl for tok in (" like ", " ilike ", " similar to ")):
        return FILTER_OP_IDS["like"]
    if " in " in cl or " = any" in cl or " = some" in cl:
        return FILTER_OP_IDS["in"]
    if " between " in cl or re.search(
        r"<=|>=|(?<![<>=!])<(?![=])|(?<![<>=!])>(?![=])", cl
    ):
        return FILTER_OP_IDS["range"]
    if "=" in cl or " is null " in cl or " is not null " in cl:
        return FILTER_OP_IDS["eq"]
    return FILTER_OP_IDS["other"]


def _extract_numeric_constant(clause: str, alias: str, col: str) -> Optional[float]:
    patterns = [
        rf"\b{re.escape(alias)}\.{re.escape(col)}\b\s*(?:=|!=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)",
        rf"(-?\d+(?:\.\d+)?)\s*(?:=|!=|<=|>=|<|>)\s*\b{re.escape(alias)}\.{re.escape(col)}\b",
    ]
    for pat in patterns:
        m = re.search(pat, clause, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    m = re.search(
        rf"\b{re.escape(alias)}\.{re.escape(col)}\b\s+BETWEEN\s+(-?\d+(?:\.\d+)?)\s+AND\s+(-?\d+(?:\.\d+)?)",
        clause,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _extract_string_constants(clause: str, alias: str, col: str) -> List[str]:
    reference = (
        rf"\b{re.escape(alias)}\.{re.escape(col)}\b"
        rf"(?:\s*\))?(?:\s*::\s*(?:\w+|\"[^\"]+\"))*"
    )
    patterns = [
        rf"{reference}\s*(?:=|!=|<>|<=|>=|<|>)\s*'((?:''|[^'])*)'",
        rf"'((?:''|[^'])*)'\s*(?:=|!=|<>|<=|>=|<|>)\s*{reference}",
    ]
    values: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, clause, re.IGNORECASE):
            values.append(match.group(1).replace("''", "'"))
    return values


def _extract_local_filter_info(
    sql: str, aliases: List[str]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    alias_set = set(a.lower() for a in aliases)
    info: Dict[str, Dict[str, Dict[str, Any]]] = {a: {} for a in alias_set}
    where_text = _extract_where_text(sql)
    if not where_text:
        return info

    clauses = _split_and_clauses(where_text)
    for clause in clauses:
        refs = {m.group(1).lower() for m in re.finditer(r"\b(\w+)\.(\w+)\b", clause)}
        refs &= alias_set
        if len(refs) != 1:
            continue
        alias = next(iter(refs))
        op_id = _classify_filter_clause(clause)
        for m in re.finditer(rf"\b{re.escape(alias)}\.(\w+)\b", clause, re.IGNORECASE):
            col = m.group(1).lower()
            rec = info[alias].setdefault(
                col,
                {"count": 0, "ops": [], "consts": [], "literals": []},
            )
            rec["count"] += 1
            rec["ops"].append(op_id)
            num_const = _extract_numeric_constant(clause, alias, col)
            if num_const is not None:
                rec["consts"].append(num_const)
            rec["literals"].extend(_extract_string_constants(clause, alias, col))
    return info


def _extract_plan_relation_stats(
    plan_json: Optional[dict],
) -> Dict[str, Tuple[float, float]]:
    if not plan_json:
        return {}
    root = plan_json.get("Plan", plan_json)
    stats: Dict[str, Tuple[float, float]] = {}

    def walk(node: dict) -> None:
        names = []
        rel = node.get("Relation Name")
        alias = node.get("Alias")
        if rel:
            names.append(str(rel).lower())
        if alias:
            names.append(str(alias).lower())
        rows = float(node.get("Plan Rows", 0))
        width = float(node.get("Plan Width", 0))
        for name in names:
            prev_rows, prev_width = stats.get(name, (0.0, 0.0))
            stats[name] = (max(prev_rows, rows), max(prev_width, width))
        for child in node.get("Plans", []):
            walk(child)

    walk(root)
    return stats


def _is_temp_relation(table_name: str) -> bool:
    tn = (table_name or "").lower()
    return tn.startswith("qs_step_") or tn in {"qs_final", "qs_prev", "qs_result"}


def _coerce_numeric_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]
        text = "".join(str(v) for v in value)
    else:
        text = str(value)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    out: List[float] = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            continue
    return out


def _parse_pg_array_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(v, str) and len(v) == 1 for v in value):
            text = "".join(value)
        else:
            return [str(v) for v in value]
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if not text:
        return []
    try:
        reader = csv.reader(
            io.StringIO(text), delimiter=",", quotechar='"', escapechar="\\"
        )
        row = next(reader, [])
        return [item.strip() for item in row if item.strip()]
    except Exception:
        return [item.strip() for item in text.split(",") if item.strip()]


def _parse_histogram_bounds(value: Any, type_family_id: int) -> List[float]:
    items = _parse_pg_array_items(value)
    if not items:
        return []
    if type_family_id == TYPE_FAMILY_IDS["numeric"]:
        out: List[float] = []
        for item in items:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    if type_family_id == TYPE_FAMILY_IDS["date"]:
        out = []
        for item in items:
            try:
                out.append(
                    float(np.datetime64(item).astype("datetime64[D]").astype(np.int64))
                )
            except Exception:
                continue
        return out
    return []


def _type_family_id(data_type: Optional[str]) -> int:
    dt = (data_type or "").lower()
    if any(
        tok in dt
        for tok in ("int", "numeric", "decimal", "real", "double", "float", "serial")
    ):
        return TYPE_FAMILY_IDS["numeric"]
    if any(tok in dt for tok in ("date", "time", "timestamp")):
        return TYPE_FAMILY_IDS["date"]
    if "bool" in dt:
        return TYPE_FAMILY_IDS["bool"]
    if any(tok in dt for tok in ("char", "text", "string", "varchar")):
        return TYPE_FAMILY_IDS["string"]
    return TYPE_FAMILY_IDS["other"]


def _normalize_ndv_ratio(n_distinct: float, rows: float) -> float:
    if rows <= 0:
        return 0.0
    nd = float(n_distinct)
    if nd < 0:
        nd = abs(nd) * rows
    return float(min(nd / max(rows, 1.0), 1.0))


def _quantile_sketch(bounds: List[float]) -> List[float]:
    if not bounds:
        return [0.0] * 5
    vals = np.asarray(bounds, dtype=np.float32)
    if vals.size == 1:
        return [float(vals[0])] * 5
    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    out = np.quantile(vals, qs).astype(np.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi > lo:
        out = (out - lo) / (hi - lo)
    else:
        out = np.zeros_like(out)
    return [float(x) for x in out]


def _cat_sketch(freqs: Sequence[float]) -> List[float]:
    vals = [float(v) for v in freqs[:4]]
    vals += [0.0] * max(0, 4 - len(vals))
    mass = min(float(sum(float(v) for v in freqs)), 1.0) if freqs else 0.0
    return vals[:4] + [mass]


def _const_position(const_values: Sequence[float], bounds: Sequence[float]) -> float:
    if not const_values or not bounds:
        return 0.0
    val = float(np.mean(const_values))
    b = np.asarray(bounds, dtype=np.float32)
    if b.size == 0:
        return 0.0
    return float(np.searchsorted(np.sort(b), val, side="right") / max(len(b), 1))


def _categorical_literal_frequency(
    literals: Sequence[str],
    most_common_values: Any,
    most_common_freqs: Sequence[float],
    n_distinct: float,
    relation_rows: float,
) -> float:
    if not literals:
        return 0.0
    values = _parse_pg_array_items(most_common_values)
    frequencies = [float(value) for value in most_common_freqs]
    lookup = {
        str(value): frequencies[index]
        for index, value in enumerate(values)
        if index < len(frequencies)
    }
    estimated_distinct = float(n_distinct)
    if estimated_distinct < 0.0:
        estimated_distinct = abs(estimated_distinct) * max(float(relation_rows), 1.0)
    fallback = 1.0 / max(estimated_distinct, 1.0) if estimated_distinct else 0.0
    return float(np.mean([lookup.get(str(literal), fallback) for literal in literals]))


def _graph_is_cyclic_undirected(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> bool:
    if n_nodes <= 2:
        return False
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * n_nodes

    def dfs(node: int, parent: int) -> bool:
        visited[node] = True
        for nxt in adj[node]:
            if not visited[nxt]:
                if dfs(nxt, node):
                    return True
            elif nxt != parent:
                return True
        return False

    for i in range(n_nodes):
        if not visited[i] and dfs(i, -1):
            return True
    return False


def build_transfer_graph_state(
    sql: str,
    graph: JoinGraph,
    catalog: CatalogInfo,
    plan_json: Optional[dict] = None,
) -> Tuple[QueryGraphState, GraphStats]:
    aliases = graph.alias_list()
    if not aliases:
        stats = GraphStats(0, 0, 0.0, 0.0, 0.0, False, 0.0)
        return empty_query_graph_state(), stats

    alias_idx = {alias: i for i, alias in enumerate(aliases)}
    plan_rel_stats = _extract_plan_relation_stats(plan_json)
    filter_info = _extract_local_filter_info(sql, aliases)
    all_refs = _extract_alias_column_refs(sql, aliases)

    join_cols: Dict[str, set] = {a: set() for a in aliases}
    filter_cols: Dict[str, set] = {
        a: set(info.keys()) for a, info in filter_info.items()
    }
    for la, ra, lc, rc in graph.edges:
        if la in join_cols:
            join_cols[la].add(lc.lower())
        if ra in join_cols:
            join_cols[ra].add(rc.lower())

    used_cols: Dict[str, List[str]] = {}
    for alias in aliases:
        cols = (
            set(all_refs.get(alias, []))
            | join_cols.get(alias, set())
            | filter_cols.get(alias, set())
        )
        used_cols[alias] = sorted(cols)

    table_feat = np.zeros((len(aliases), TABLE_NODE_FEAT_DIM), dtype=np.float32)
    column_rows: List[np.ndarray] = []
    membership_edges: List[Tuple[int, int]] = []
    column_lookup: Dict[Tuple[str, str], int] = {}

    for alias in aliases:
        table = graph.aliases[alias]
        table_i = alias_idx[alias]
        table_cols = catalog.table_columns(table)
        indexed_cols = [c for c in table_cols if catalog.has_index(table, c)]
        rel_rows = float(catalog.table_row_count(table))
        plan_rows = 0.0
        for key in (table, alias):
            plan_rows = max(plan_rows, plan_rel_stats.get(key, (0.0, 0.0))[0])
        feature_rows = plan_rows if plan_rows > 0 else rel_rows
        if _is_temp_relation(table) and plan_rows > 0:
            rel_rows = plan_rows

        table_feat[table_i, 0] = np.log1p(feature_rows) / 18.0
        table_feat[table_i, 1] = min(len(used_cols[alias]), 16) / 16.0
        table_feat[table_i, 2] = min(len(join_cols.get(alias, set())), 8) / 8.0
        table_feat[table_i, 3] = min(len(filter_cols.get(alias, set())), 8) / 8.0
        table_feat[table_i, 4] = (
            len(indexed_cols) / max(len(table_cols), 1) if table_cols else 0.0
        )

        for col in used_cols[alias]:
            cs = np.zeros(COLUMN_NODE_FEAT_DIM, dtype=np.float32)
            type_family_id = _type_family_id(catalog.col_data_type(table, col))
            cs[COLUMN_IDX_TYPE_FAMILY] = float(type_family_id)
            cs[COLUMN_IDX_NULL_FRAC] = float(catalog.col_null_frac(table, col))
            cs[COLUMN_IDX_NDV_RATIO] = _normalize_ndv_ratio(
                catalog.col_n_distinct(table, col), rel_rows
            )
            cs[COLUMN_IDX_AVG_WIDTH] = min(
                float(catalog.col_avg_width(table, col)) / 128.0, 1.0
            )
            cs[COLUMN_IDX_KEY_ROLE] = float(catalog.key_role_id(table, col))
            cs[COLUMN_IDX_HAS_INDEX] = 1.0 if catalog.has_index(table, col) else 0.0

            rec = filter_info.get(alias, {}).get(col, {})
            count = int(rec.get("count", 0))
            cs[COLUMN_IDX_NUM_FILTERS] = min(count, 4) / 4.0
            ops = rec.get("ops", [])
            if ops:
                counts: Dict[int, int] = {}
                for op in ops:
                    counts[op] = counts.get(op, 0) + 1
                dominant = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            else:
                dominant = FILTER_OP_IDS["none"]
            cs[COLUMN_IDX_FILTER_OP] = float(dominant)

            bounds = _parse_histogram_bounds(
                catalog.col_histogram_bounds(table, col), type_family_id
            )
            if type_family_id in {
                TYPE_FAMILY_IDS["numeric"],
                TYPE_FAMILY_IDS["date"],
            }:
                cs[COLUMN_IDX_CONST_POS] = _const_position(
                    rec.get("consts", []), bounds
                )
            else:
                cs[COLUMN_IDX_CONST_POS] = _categorical_literal_frequency(
                    rec.get("literals", []),
                    catalog.col_most_common_values(table, col),
                    catalog.col_most_common_freqs(table, col),
                    catalog.col_n_distinct(table, col),
                    rel_rows,
                )
            cs[COLUMN_IDX_NUM_DIST] = np.asarray(
                _quantile_sketch(bounds), dtype=np.float32
            )
            cs[COLUMN_IDX_CAT_DIST] = np.asarray(
                _cat_sketch(
                    _coerce_numeric_list(catalog.col_most_common_freqs(table, col))
                ),
                dtype=np.float32,
            )

            col_idx = len(column_rows)
            column_rows.append(cs)
            membership_edges.append((table_i, col_idx))
            column_lookup[(alias, col)] = col_idx

    if not column_rows:
        column_rows.append(np.zeros(COLUMN_NODE_FEAT_DIM, dtype=np.float32))
        membership_edges.append((0, 0))

    join_edges: List[Tuple[int, int]] = []
    join_edge_feat: List[np.ndarray] = []
    table_pair_counts: Dict[Tuple[int, int], int] = {}

    for la, ra, lc, rc in graph.edges:
        lkey = (la, lc.lower())
        rkey = (ra, rc.lower())
        if lkey not in column_lookup or rkey not in column_lookup:
            continue
        li = column_lookup[lkey]
        ri = column_lookup[rkey]
        join_edges.append((li, ri))
        lt = graph.aliases.get(la, la)
        rt = graph.aliases.get(ra, ra)
        l_is_fk = catalog.is_fk(lt, lc)
        r_is_fk = catalog.is_fk(rt, rc)
        l_is_pk = catalog.is_pk(lt, lc)
        r_is_pk = catalog.is_pk(rt, rc)
        fk_pk_flag = 1.0 if ((l_is_fk and r_is_pk) or (r_is_fk and l_is_pk)) else 0.0
        join_edge_feat.append(
            np.asarray([float(JOIN_OP_IDS["eq"]), fk_pk_flag], dtype=np.float32)
        )
        ti, tj = alias_idx[la], alias_idx[ra]
        pair = (min(ti, tj), max(ti, tj))
        table_pair_counts[pair] = table_pair_counts.get(pair, 0) + 1

    table_join_edges = np.asarray(list(table_pair_counts.keys()), dtype=np.int64)
    table_join_feat = np.zeros(
        (len(table_pair_counts), TABLE_JOIN_EDGE_FEAT_DIM), dtype=np.float32
    )
    for idx, pair in enumerate(table_pair_counts):
        table_join_feat[idx, 0] = min(table_pair_counts[pair], 8) / 8.0

    n_tables = len(aliases)
    table_edges = list(table_pair_counts.keys())
    degrees = np.zeros(n_tables, dtype=np.float32)
    for u, v in table_edges:
        if u == v:
            continue
        degrees[u] += 1
        degrees[v] += 1
    n_edges = len(table_edges)
    avg_degree = float((2.0 * n_edges) / max(n_tables, 1))
    edge_density = float((2.0 * n_edges) / max(n_tables * max(n_tables - 1, 1), 1))
    is_cyclic = _graph_is_cyclic_undirected(n_tables, table_edges)
    star_score = (
        float(min(max(degrees.max(initial=0.0) / max(n_edges, 1), 0.0), 1.0))
        if n_edges > 0
        else 0.0
    )
    stats = GraphStats(
        n_nodes=n_tables,
        n_edges=n_edges,
        max_degree=float(degrees.max(initial=0.0)) if degrees.size else 0.0,
        avg_degree=avg_degree,
        edge_density=edge_density,
        is_cyclic=is_cyclic,
        star_score=star_score,
    )

    qgraph = QueryGraphState(
        table_node_features=table_feat,
        column_node_features=np.vstack(column_rows).astype(np.float32),
        membership_edges=np.asarray(membership_edges, dtype=np.int64),
        join_edges=(
            np.asarray(join_edges, dtype=np.int64)
            if join_edges
            else np.zeros((0, 2), dtype=np.int64)
        ),
        join_edge_features=(
            np.vstack(join_edge_feat).astype(np.float32)
            if join_edge_feat
            else np.zeros((0, JOIN_EDGE_FEAT_DIM), dtype=np.float32)
        ),
        table_join_edges=(
            table_join_edges
            if len(table_pair_counts)
            else np.zeros((0, 2), dtype=np.int64)
        ),
        table_join_edge_features=(
            table_join_feat
            if len(table_pair_counts)
            else np.zeros((0, TABLE_JOIN_EDGE_FEAT_DIM), dtype=np.float32)
        ),
    )
    return qgraph, stats


def classify_plan_op(node_type: str) -> int:
    nt = (node_type or "").lower()
    if "seq scan" in nt:
        return PLAN_OP_TYPES.index("seq_scan")
    if "index only scan" in nt or "index scan" in nt:
        return PLAN_OP_TYPES.index("index_scan")
    if "bitmap" in nt:
        return PLAN_OP_TYPES.index("bitmap_scan")
    if "hash join" in nt:
        return PLAN_OP_TYPES.index("hash_join")
    if "nested loop" in nt:
        return PLAN_OP_TYPES.index("nested_loop")
    if "merge join" in nt:
        return PLAN_OP_TYPES.index("merge_join")
    if "sort" in nt:
        return PLAN_OP_TYPES.index("sort")
    if "aggregate" in nt:
        return PLAN_OP_TYPES.index("aggregate")
    if nt == "hash" or nt.endswith(" hash"):
        return PLAN_OP_TYPES.index("hash")
    if "materialize" in nt:
        return PLAN_OP_TYPES.index("materialize")
    if "append" in nt:
        return PLAN_OP_TYPES.index("append")
    if "limit" in nt:
        return PLAN_OP_TYPES.index("limit")
    return PLAN_OTHER_IDX


def _tree_max_depth(node: dict, depth: int = 0) -> int:
    children = node.get("Plans", [])
    if not children:
        return depth
    return max(_tree_max_depth(child, depth + 1) for child in children)


def plan_to_tree(
    plan_json: Optional[dict], catalog: Optional[CatalogInfo] = None
) -> PlanTree:
    if plan_json is None:
        return empty_plan_tree()

    root = plan_json.get("Plan", plan_json)
    root_plan_rows = max(float(root.get("Plan Rows", 0)), 1.0)
    max_depth = max(_tree_max_depth(root), 1)

    def base_rel_rows(node: dict) -> float:
        rel = str(node.get("Relation Name", "") or "").lower()
        if not rel or catalog is None:
            return 0.0
        return float(catalog.table_row_count(rel))

    def build(node: dict, depth: int = 0) -> PlanTree:
        children = tuple(build(child, depth + 1) for child in node.get("Plans", []))
        plan_rows = float(node.get("Plan Rows", 0))
        rel_rows = base_rel_rows(node)
        scan_sel_est = min(plan_rows / max(rel_rows, 1.0), 1.0) if rel_rows > 0 else 0.0

        feat = np.zeros(PLAN_NODE_FEAT_DIM, dtype=np.float32)
        feat[0] = np.log1p(float(node.get("Total Cost", 0))) / 15.0
        feat[1] = np.log1p(plan_rows) / 15.0
        feat[2] = scan_sel_est
        feat[3] = min(plan_rows / root_plan_rows, 1.0)
        feat[4] = np.log1p(float(node.get("Plan Width", 0))) / 10.0
        feat[5] = depth / max(max_depth, 1)
        feat[6] = min(len(children), 4) / 4.0
        feat[7] = 1.0 if node.get("Filter") else 0.0
        feat[8] = 1.0 if node.get("Index Cond") else 0.0
        feat[9] = 1.0 if node.get("Recheck Cond") else 0.0
        return PlanTree(
            op_type_id=classify_plan_op(node.get("Node Type", "")),
            node_numeric_feat=feat,
            children=children,
            is_sentinel=False,
        )

    return build(root)


class TableFeatureEncoder(nn.Module):
    def __init__(self, out_dim: int = GRAPH_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TABLE_NODE_FEAT_DIM, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ColumnFeatureEncoder(nn.Module):
    def __init__(self, out_dim: int = GRAPH_EMB_DIM):
        super().__init__()
        self.type_embed = nn.Embedding(len(TYPE_FAMILY_IDS), 8)
        self.key_embed = nn.Embedding(len(KEY_ROLE_IDS), 6)
        self.filter_embed = nn.Embedding(len(FILTER_OP_IDS), 6)
        self.scalar_proj = nn.Sequential(
            nn.Linear(16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(out_dim + 20, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type_ids = (
            x[:, COLUMN_IDX_TYPE_FAMILY]
            .long()
            .clamp_min(0)
            .clamp_max(len(TYPE_FAMILY_IDS) - 1)
        )
        key_ids = (
            x[:, COLUMN_IDX_KEY_ROLE]
            .long()
            .clamp_min(0)
            .clamp_max(len(KEY_ROLE_IDS) - 1)
        )
        filter_ids = (
            x[:, COLUMN_IDX_FILTER_OP]
            .long()
            .clamp_min(0)
            .clamp_max(len(FILTER_OP_IDS) - 1)
        )
        scalar = torch.cat(
            [
                x[
                    :,
                    [
                        COLUMN_IDX_NULL_FRAC,
                        COLUMN_IDX_NDV_RATIO,
                        COLUMN_IDX_AVG_WIDTH,
                        COLUMN_IDX_HAS_INDEX,
                        COLUMN_IDX_NUM_FILTERS,
                        COLUMN_IDX_CONST_POS,
                    ],
                ],
                x[:, COLUMN_IDX_NUM_DIST],
                x[:, COLUMN_IDX_CAT_DIST],
            ],
            dim=1,
        )
        scalar_h = self.scalar_proj(scalar)
        cat_h = torch.cat(
            [
                self.type_embed(type_ids),
                self.key_embed(key_ids),
                self.filter_embed(filter_ids),
            ],
            dim=1,
        )
        return self.out(torch.cat([scalar_h, cat_h], dim=1))


class JoinEdgeEncoder(nn.Module):
    def __init__(self, out_dim: int = GRAPH_EMB_DIM):
        super().__init__()
        self.join_op_embed = nn.Embedding(len(JOIN_OP_IDS), 8)
        self.net = nn.Sequential(
            nn.Linear(9, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        join_op = feat[:, 0].long().clamp_min(0).clamp_max(len(JOIN_OP_IDS) - 1)
        fk_flag = feat[:, 1:2]
        return self.net(torch.cat([self.join_op_embed(join_op), fk_flag], dim=1))


class TableJoinEdgeEncoder(nn.Module):
    def __init__(self, out_dim: int = GRAPH_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TABLE_JOIN_EDGE_FEAT_DIM, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def _aggregate_messages(
    num_dst: int,
    edges: torch.Tensor,
    src_h: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    edge_h: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edges.numel() == 0 or num_dst == 0:
        return torch.zeros((num_dst, src_h.size(1) * 2), device=src_h.device)
    dim = src_h.size(1)
    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_dst)]
    for eidx in range(edges.size(0)):
        s = int(src_indices[eidx].item())
        d = int(dst_indices[eidx].item())
        msg = src_h[s]
        if edge_h is not None and edge_h.size(0) > eidx:
            msg = msg + edge_h[eidx]
        buckets[d].append(msg)

    zeros = torch.zeros(dim, device=src_h.device)
    means: List[torch.Tensor] = []
    maxes: List[torch.Tensor] = []
    for msgs in buckets:
        if not msgs:
            means.append(zeros)
            maxes.append(zeros)
            continue
        stack = torch.stack(msgs, dim=0)
        means.append(stack.mean(dim=0))
        maxes.append(stack.max(dim=0).values)
    return torch.cat([torch.stack(means, dim=0), torch.stack(maxes, dim=0)], dim=1)


class HeteroGraphLayer(nn.Module):
    def __init__(self, dim: int = GRAPH_EMB_DIM):
        super().__init__()
        self.join_edge_encoder = JoinEdgeEncoder(dim)
        self.table_join_edge_encoder = TableJoinEdgeEncoder(dim)
        self.table_update = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
        self.column_update = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

    def forward(
        self,
        table_h: torch.Tensor,
        column_h: torch.Tensor,
        qgraph: QueryGraphState,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mem_edges = qgraph.tensor(
            "membership_edges", qgraph.membership_edges, device, torch.long
        )
        join_edges = qgraph.tensor("join_edges", qgraph.join_edges, device, torch.long)
        join_feat = qgraph.tensor(
            "join_edge_features", qgraph.join_edge_features, device
        )
        table_join_edges = qgraph.tensor(
            "table_join_edges", qgraph.table_join_edges, device, torch.long
        )
        table_join_feat = qgraph.tensor(
            "table_join_edge_features", qgraph.table_join_edge_features, device
        )

        # column -> table
        col_to_table = _aggregate_messages(
            table_h.size(0),
            mem_edges,
            column_h,
            mem_edges[:, 1] if mem_edges.numel() else mem_edges,
            mem_edges[:, 0] if mem_edges.numel() else mem_edges,
            None,
        )

        # table -> column (each column belongs to exactly one table)
        table_to_col_list: List[Optional[torch.Tensor]] = [None] * column_h.size(0)
        if mem_edges.numel():
            for eidx in range(mem_edges.size(0)):
                t = int(mem_edges[eidx, 0].item())
                c = int(mem_edges[eidx, 1].item())
                msg = torch.cat([table_h[t], table_h[t]], dim=0)
                table_to_col_list[c] = msg
        zero_col_msg = torch.zeros(table_h.size(1) * 2, device=device)
        table_to_col = torch.stack(
            [msg if msg is not None else zero_col_msg for msg in table_to_col_list],
            dim=0,
        )

        # column <-> column join messages
        join_h = (
            self.join_edge_encoder(join_feat)
            if join_feat.numel()
            else torch.zeros((0, table_h.size(1)), device=device)
        )
        col_join = torch.zeros((column_h.size(0), column_h.size(1) * 2), device=device)
        if join_edges.numel():
            col_join_fwd = _aggregate_messages(
                column_h.size(0),
                join_edges,
                column_h,
                join_edges[:, 0],
                join_edges[:, 1],
                join_h,
            )
            col_join_rev = _aggregate_messages(
                column_h.size(0),
                join_edges,
                column_h,
                join_edges[:, 1],
                join_edges[:, 0],
                join_h,
            )
            col_join = col_join_fwd + col_join_rev

        # table <-> table shortcut messages
        table_join_h = (
            self.table_join_edge_encoder(table_join_feat)
            if table_join_feat.numel()
            else torch.zeros((0, table_h.size(1)), device=device)
        )
        table_from_tables = torch.zeros(
            (table_h.size(0), table_h.size(1) * 2), device=device
        )
        if table_join_edges.numel():
            table_from_tables_fwd = _aggregate_messages(
                table_h.size(0),
                table_join_edges,
                table_h,
                table_join_edges[:, 0],
                table_join_edges[:, 1],
                table_join_h,
            )
            table_from_tables_rev = _aggregate_messages(
                table_h.size(0),
                table_join_edges,
                table_h,
                table_join_edges[:, 1],
                table_join_edges[:, 0],
                table_join_h,
            )
            table_from_tables = table_from_tables_fwd + table_from_tables_rev

        new_table = self.table_update(
            torch.cat([table_h, col_to_table, table_from_tables], dim=1)
        )
        new_col = self.column_update(
            torch.cat([column_h, table_to_col, col_join], dim=1)
        )
        return new_table, new_col


class HeteroQueryGraphEncoder(nn.Module):
    def __init__(self, out_dim: int = GRAPH_EMB_DIM, n_layers: int = 2):
        super().__init__()
        self.table_encoder = TableFeatureEncoder(out_dim)
        self.column_encoder = ColumnFeatureEncoder(out_dim)
        self.layers = nn.ModuleList(
            [HeteroGraphLayer(out_dim) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, qgraph: QueryGraphState, device: torch.device) -> torch.Tensor:
        table_x = qgraph.tensor(
            "table_node_features", qgraph.table_node_features, device
        )
        column_x = qgraph.tensor(
            "column_node_features", qgraph.column_node_features, device
        )
        table_h = self.table_encoder(table_x)
        column_h = self.column_encoder(column_x)

        for layer in self.layers:
            table_h, column_h = layer(table_h, column_h, qgraph, device)

        pooled = torch.cat([table_h.mean(dim=0), table_h.max(dim=0).values], dim=0)
        return self.readout(pooled)


class PlanTreeEncoder(nn.Module):
    """Lightweight TreeCNN-style encoder for PostgreSQL plan trees."""

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.ff(self.norm(x))

    def __init__(self, out_dim: int = PLAN_EMB_DIM, op_dim: int = 12):
        super().__init__()
        self.out_dim = out_dim
        self.op_embed = nn.Embedding(len(PLAN_OP_TYPES), op_dim)
        self.empty_embedding = nn.Parameter(torch.zeros(out_dim))
        self.node_base = nn.Linear(op_dim + PLAN_NODE_FEAT_DIM, out_dim)
        self.self_proj = nn.Linear(out_dim, out_dim)
        self.child_mean_proj = nn.Linear(out_dim, out_dim)
        self.child_max_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.block = self.ResidualBlock(out_dim)

    def encode_tree(self, tree: PlanTree, device: torch.device) -> torch.Tensor:
        if tree is None or tree.is_sentinel:
            return self.empty_embedding.to(device)

        op_emb = self.op_embed(tree.op_tensor(device))
        feat = tree.feat_tensor(device)
        node_base = self.node_base(torch.cat([op_emb, feat], dim=0))

        child_embs = [self.encode_tree(child, device) for child in tree.children]
        if child_embs:
            child_stack = torch.stack(child_embs, dim=0)
            child_mean = child_stack.mean(dim=0)
            child_max = child_stack.max(dim=0).values
        else:
            child_mean = torch.zeros(self.out_dim, device=device)
            child_max = torch.zeros(self.out_dim, device=device)

        x = (
            self.self_proj(node_base)
            + self.child_mean_proj(child_mean)
            + self.child_max_proj(child_max)
        )
        x = F.silu(self.norm(x))
        x = x + 0.5 * child_mean
        return self.block(x)

    def forward_batch(
        self, trees: List[PlanTree], device: torch.device
    ) -> torch.Tensor:
        return torch.stack([self.encode_tree(tree, device) for tree in trees], dim=0)


class BaseNetworkTransfer(nn.Module):
    """Single supported encoder: table+column hetero graph + TreeCNN plan encoder."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        lite_hidden = min(hidden, 96)
        self.hidden_out = lite_hidden
        self.graph_encoder = HeteroQueryGraphEncoder(GRAPH_EMB_DIM, n_layers=2)
        self.plan_encoder = PlanTreeEncoder(PLAN_EMB_DIM)
        self.dec_trunk = self._make_trunk(DEC_STATE_DIM, lite_hidden)
        self.enum_trunk = self._make_trunk(ENUM_STATE_DIM, lite_hidden)
        self.adapt_trunk = self._make_trunk(ADAPT_STATE_DIM, lite_hidden)

    @staticmethod
    def _make_trunk(input_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

    def encode_structured_state(
        self,
        state: StructuredState,
        device: torch.device,
        *,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        if state.level == "adapt":
            current_plan_emb = self.plan_encoder.encode_tree(
                state.current_plan,
                device,
            )
            if detach_shared:
                current_plan_emb = current_plan_emb.detach()
            ctx = state.tensor("ctx", state.ctx, device)
            return self.adapt_trunk(torch.cat([current_plan_emb, ctx], dim=0))

        graph_emb = self.graph_encoder(state.query_graph, device)
        if detach_shared:
            graph_emb = graph_emb.detach()
        if state.level == "dec":
            ctx = state.tensor("ctx", state.ctx, device)
            fused = torch.cat([graph_emb, ctx], dim=0)
            return self.dec_trunk(fused)
        return self.enum_trunk(graph_emb)

    def encode_structured_batch(
        self,
        states: List[StructuredState],
        device: torch.device,
        *,
        detach_shared: bool = False,
    ) -> torch.Tensor:
        return _encode_unique_structured_batch(
            states,
            device,
            lambda state, target_device: self.encode_structured_state(
                state,
                target_device,
                detach_shared=detach_shared,
            ),
        )
