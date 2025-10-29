from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
import json
import hashlib

PlanDict = Dict[str, Any]


def _normalize_root(root: Union[List[PlanDict], PlanDict]) -> PlanDict:
    if isinstance(root, list):
        # EXPLAIN JSON is usually a 1-element list
        root = root[0] if root else {}
    return root.get("Plan", root)


def _children(node: PlanDict) -> Iterable[PlanDict]:
    # Normal plan children
    if "Plans" in node:
        for c in node["Plans"]:
            yield c
    # Workers may carry their own Plans
    if "Workers" in node:
        for w in node["Workers"]:
            if "Plans" in w:
                for c in w["Plans"]:
                    yield c
    # InitPlans (subqueries executed before the node)
    if "InitPlan" in node:
        for c in node["InitPlan"]:
            yield c
    # CTEs appear at the top level; scans show as "CTE Scan"
    # Nothing to yield here; scans will be covered above.


def _is_scan(node: PlanDict) -> bool:
    nt = node.get("Node Type", "")
    return nt.endswith("Scan") or nt in ("CTE Scan", "Subquery Scan")


def _node_id(node: PlanDict) -> str:
    rel = node.get("Relation Name")
    alias = node.get("Alias")
    idx = node.get("Index Name")
    if alias and rel:
        return f"{alias}({rel})"
    return alias or rel or idx or node.get("Node Type", "Unknown")


def _inclusive_time(node: PlanDict) -> Optional[float]:
    t = node.get("Actual Total Time")
    loops = node.get("Actual Loops", 1)
    if t is None:
        return None
    try:
        return float(t) * float(loops or 1)
    except Exception:
        return None


class PlanHelper:
    @classmethod
    def extract_plan_info(cls, plan: Union[List[PlanDict], PlanDict]) -> Dict[str, Any]:
        """Structured plan info; robust to arrays/workers/initplans."""
        root = _normalize_root(plan)

        def helper(node: PlanDict) -> Dict[str, Any]:
            info = {
                "node_type": node.get("Node Type"),
                "relation_name": node.get("Relation Name"),
                "alias": node.get("Alias"),
                "index_name": node.get("Index Name"),
                "join_type": node.get("Join Type"),
                "actual_rows": node.get("Actual Rows"),
                "actual_loops": node.get("Actual Loops"),
                "actual_startup_time": node.get("Actual Startup Time"),
                "actual_total_time": node.get("Actual Total Time"),
                "plan_rows": node.get("Plan Rows"),
                "plan_width": node.get("Plan Width"),
                "startup_cost": node.get("Startup Cost"),
                "total_cost": node.get("Total Cost"),
                "subplans": [],
            }
            children = list(_children(node))
            if children:
                info["subplans"] = [helper(c) for c in children]
            return info

        return helper(root)

    @classmethod
    def extract_join_order(cls, plan: Union[List[PlanDict], PlanDict]) -> List[str]:
        """Base-table join order (pre-order over joins; scans only)."""
        root = _normalize_root(plan)
        order: List[str] = []
        seen: set = set()

        def pre(node: PlanDict):
            # Append scan nodes immediately; for joins, we still dive left-to-right
            if _is_scan(node):
                nid = _node_id(node)
                if nid and nid not in seen:
                    seen.add(nid)
                    order.append(nid)
            for c in _children(node):
                pre(c)

        pre(root)
        return order

    @classmethod
    def extract_operator_sequence(cls, plan: Union[List[PlanDict], PlanDict],
                                  order: str = "pre") -> List[str]:
        """Operator sequence. order ∈ {'pre','post'}. """
        root = _normalize_root(plan)
        out: List[str] = []

        def pre(node: PlanDict):
            nt = node.get("Node Type")
            if nt: out.append(nt)
            for c in _children(node): pre(c)

        def post(node: PlanDict):
            for c in _children(node): post(c)
            nt = node.get("Node Type")
            if nt: out.append(nt)

        (pre if order == "pre" else post)(root)
        return out

    @classmethod
    def extract_subplan_latencies(cls, plan: Union[List[PlanDict], PlanDict],
                                  mode: str = "inclusive") -> Dict[str, float]:
        """
        Subplan latencies with loop-aware timing.
        mode: 'inclusive' or 'exclusive'
        """
        root = _normalize_root(plan)
        path_times: Dict[str, float] = {}

        def walk(node: PlanDict, path: str) -> float:
            name = node.get("Node Type", "Unknown")
            ident = _node_id(node)
            label = f"{path}/{name}[{ident}]" if path else f"{name}[{ident}]"
            child_sum = 0.0
            for c in _children(node):
                child_sum += walk(c, label)
            t = _inclusive_time(node) or 0.0
            total = t
            if mode == "exclusive":
                total = max(0.0, t - child_sum)
            path_times[label] = total
            return t  # return inclusive to accumulate to parents

        walk(root, "")
        return path_times

    @classmethod
    def compute_plan_hash(cls, plan_json: Dict[str, Any]) -> str:
        """Compute a hash for the plan structure (ignoring timing)"""

        # Normalize EXPLAIN JSON root (list -> dict and unwrap to "Plan")
        root = _normalize_root(plan_json)

        def extract_structure(node):
            """Extract structural fields (physical operators and topology only).
            Excludes volatile numeric fields like costs/rows/timing."""
            if not isinstance(node, dict):
                return node
            structure = {
                'node_type': node.get('Node Type'),
                'relation_name': node.get('Relation Name'),
                'alias': node.get('Alias'),
                'index_name': node.get('Index Name'),
                'join_type': node.get('Join Type'),
            }
            # Collect children through unified iterator (_children handles Plans/Workers/InitPlan)
            children = list(_children(node))
            if children:
                structure['plans'] = [extract_structure(child) for child in children]
                return structure

        structure = extract_structure(root)
        structure_str = json.dumps(structure, sort_keys=True)
        return hashlib.md5(structure_str.encode()).hexdigest()
