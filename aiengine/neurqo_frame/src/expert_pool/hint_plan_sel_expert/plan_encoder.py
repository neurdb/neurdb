import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


@dataclass
class PlanNode:
    """Represents a node in the query plan tree"""

    node_type: str
    relation_name: Optional[str] = None
    index_name: Optional[str] = None
    alias: Optional[str] = None
    total_cost: Optional[float] = None
    plan_rows: Optional[float] = None
    buffers: Optional[float] = None
    children: List["PlanNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class StatExtractor:
    """Helper class for extracting and normalizing plan statistics."""

    def __init__(self, fields: List[str], mins: List[float], maxs: List[float]):
        self.fields = fields
        self.mins = mins
        self.maxs = maxs

    def __call__(self, inp: Dict[str, Any]) -> List[float]:
        """Extract and normalize statistics from a node."""
        res = []
        for f, lo, hi in zip(self.fields, self.mins, self.maxs):
            if f not in inp:
                res.append(0)
            else:
                res.append(self._norm(inp[f], lo, hi))
        return res

    def _norm(self, x: float, lo: float, hi: float) -> float:
        """Normalize a value using log transformation."""
        return (np.log(x + 1) - lo) / (hi - lo)


class OneHotPlanEncoder:
    """
    Encoder for query plans based on Bao's TreeFeaturizer.
    Converts PostgreSQL EXPLAIN ANALYZE JSON plans into feature trees for neural network training.
    """

    # Operator types from Bao
    JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
    LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
    ALL_TYPES = JOIN_TYPES + LEAF_TYPES

    def __init__(self):
        super().__init__()
        self.stats_extractor = None
        self.relations = None
        self.is_fitted = False

    def fit(self, plans: List[Dict[str, Any]]):
        """
        Fit the encoder on a set of plans to learn normalization parameters.

        Args:
            plans: List of plan JSON dictionaries from EXPLAIN ANALYZE
        """
        # Extract all relations from the plans
        self.relations = self._get_all_relations(plans)

        # Create stats extractor for normalization
        self.stats_extractor = self._get_plan_stats(plans)

        self.is_fitted = True

    def transform(self, plans: List[Dict[str, Any]]) -> List[Tuple]:
        """
        Transform plans into feature trees.

        Args:
            plans: List of plan JSON dictionaries

        Returns:
            List of feature trees (tuples) ready for neural network input
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Attach buffer data to plans
        for plan in plans:
            self._attach_buffer_data(plan)

        # Convert to feature trees
        trees = []
        for plan in plans:
            tree = self._plan_to_feature_tree(plan)
            trees.append(tree)

        return trees

    def num_operators(self) -> int:
        """Get the number of operator types."""
        return len(self.ALL_TYPES)

    def get_feature_dimension(self) -> Optional[int]:
        """Get the dimension of the feature representation."""
        if self.is_fitted:
            # Estimate based on operator types and relation features
            return len(self.ALL_TYPES) + (len(self.relations) if self.relations else 0)
        return None

    def save(self, path: str):
        """
        Save the encoder state to disk.

        Args:
            path: Directory path to save the encoder state
        """
        os.makedirs(path, exist_ok=True)

        # Save encoder state
        encoder_state = {
            "is_fitted": self.is_fitted,
            "relations": self.relations,
            "join_types": self.JOIN_TYPES,
            "leaf_types": self.LEAF_TYPES,
            "all_types": self.ALL_TYPES,
        }

        with open(os.path.join(path, "encoder_state.json"), "w") as f:
            json.dump(encoder_state, f, default=str)

        # Save stats extractor if it exists
        if self.stats_extractor is not None:
            with open(os.path.join(path, "stats_extractor.pkl"), "wb") as f:
                joblib.dump(self.stats_extractor, f)

    def load(self, path: str):
        """
        Load the encoder state from disk.

        Args:
            path: Directory path containing the saved encoder state
        """
        # Load encoder state
        encoder_state_path = os.path.join(path, "encoder_state.json")
        assert os.path.exists(encoder_state_path)
        with open(encoder_state_path, "r") as f:
            encoder_state = json.load(f)

        self.is_fitted = encoder_state["is_fitted"]
        self.relations = encoder_state["relations"]

        # Load stats extractor if it exists
        stats_extractor_path = os.path.join(path, "stats_extractor.pkl")
        assert os.path.exists(stats_extractor_path)
        with open(stats_extractor_path, "rb") as f:
            self.stats_extractor = joblib.load(f)

    def _get_all_relations(self, plans: List[Dict[str, Any]]):
        """Extract all relation names from plans."""
        all_rels = set()

        def recurse(plan):
            if "Relation Name" in plan:
                yield plan["Relation Name"]
            if "Plans" in plan:
                for child in plan["Plans"]:
                    yield from recurse(child)

        for plan in plans:
            all_rels.update(recurse(plan))

        return sorted(all_rels, key=lambda x: len(x), reverse=True)

    def _get_plan_stats(self, plans: List[Dict[str, Any]]):
        """Create statistics extractor for normalization."""
        costs = []
        rows = []
        bufs = []

        def recurse(n, buffers=None):
            costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            if "Buffers" in n:
                bufs.append(n["Buffers"])
            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child, buffers)

        for plan in plans:
            recurse(plan, buffers=plan.get("Buffers", None))

        costs = np.array(costs)
        rows = np.array(rows)
        bufs = np.array(bufs) if bufs else np.array([])

        # Log transform
        costs = np.log(costs + 1)
        rows = np.log(rows + 1)
        bufs = np.log(bufs + 1) if len(bufs) > 0 else np.array([])

        # Compute min/max for normalization
        costs_min, costs_max = np.min(costs), np.max(costs)
        rows_min, rows_max = np.min(rows), np.max(rows)
        bufs_min = np.min(bufs) if len(bufs) > 0 else 0
        bufs_max = np.max(bufs) if len(bufs) > 0 else 0

        if len(bufs) > 0:
            return StatExtractor(
                ["Buffers", "Total Cost", "Plan Rows"],
                [bufs_min, costs_min, rows_min],
                [bufs_max, costs_max, rows_max],
            )
        else:
            return StatExtractor(
                ["Total Cost", "Plan Rows"],
                [costs_min, rows_min],
                [costs_max, rows_max],
            )

    def _attach_buffer_data(self, plan: Dict[str, Any]):
        """Attach buffer information to leaf nodes."""
        if "Buffers" not in plan:
            return

        buffers = plan["Buffers"]

        def recurse(n):
            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)
                return

            # It's a leaf node
            n["Buffers"] = self._get_buffer_count_for_leaf(n, buffers)

        recurse(plan["Plan"])

    def _get_buffer_count_for_leaf(
        self, leaf: Dict[str, Any], buffers: Dict[str, int]
    ) -> int:
        """Get buffer count for a leaf node."""
        total = 0
        if "Relation Name" in leaf:
            total += buffers.get(leaf["Relation Name"], 0)
        if "Index Name" in leaf:
            total += buffers.get(leaf["Index Name"], 0)
        return total

    def _plan_to_feature_tree(self, plan: Dict[str, Any]) -> Tuple:
        """Convert a plan to a feature tree."""
        children = plan.get("Plans", [])

        # Handle transparent nodes (single child)
        if len(children) == 1:
            return self._plan_to_feature_tree(children[0])

        # Handle joins
        if self._is_join(plan):
            assert len(children) == 2
            my_vec = self._featurize_join(plan)
            left = self._plan_to_feature_tree(children[0])
            right = self._plan_to_feature_tree(children[1])
            return (my_vec, left, right)

        # Handle scans
        if self._is_scan(plan):
            assert not children
            return self._featurize_scan(plan)

        raise ValueError(f"Node wasn't transparent, a join, or a scan: {plan}")

    def _is_join(self, node: Dict[str, Any]) -> bool:
        """Check if node is a join operator."""
        return node["Node Type"] in self.JOIN_TYPES

    def _is_scan(self, node: Dict[str, Any]) -> bool:
        """Check if node is a scan operator."""
        return node["Node Type"] in self.LEAF_TYPES

    def _featurize_join(self, node: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for a join node."""
        assert self._is_join(node)
        arr = np.zeros(len(self.ALL_TYPES))
        arr[self.ALL_TYPES.index(node["Node Type"])] = 1
        return np.concatenate((arr, self.stats_extractor(node)))

    def _featurize_scan(self, node: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Create feature vector for a scan node."""
        assert self._is_scan(node)
        arr = np.zeros(len(self.ALL_TYPES))
        arr[self.ALL_TYPES.index(node["Node Type"])] = 1
        relation_name = self._get_relation_name(node)
        return (np.concatenate((arr, self.stats_extractor(node))), relation_name)

    def _get_relation_name(self, node: Dict[str, Any]) -> str:
        """Extract relation name from a node."""
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # Find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                raise ValueError(
                    "Bitmap operator did not have an index name or a relation name"
                )

            for rel in self.relations:
                if rel in node[name_key]:
                    return rel

            raise ValueError("Could not find relation name for bitmap index scan")

        raise ValueError("Cannot extract relation type from node")


def encode_plan_from_json(plan_json: str) -> Dict[str, Any]:
    """
    Utility function to encode a plan from JSON string.

    Args:
        plan_json: JSON string of EXPLAIN ANALYZE output

    Returns:
        Dictionary with encoded plan information
    """
    if isinstance(plan_json, str):
        plan = json.loads(plan_json)
    else:
        plan = plan_json

    return {
        "plan": plan,
        "plan_json": plan_json if isinstance(plan_json, str) else json.dumps(plan_json),
    }
