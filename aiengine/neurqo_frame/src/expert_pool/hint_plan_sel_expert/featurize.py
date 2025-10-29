import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


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
    children: List['PlanNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def norm(x, lo, hi):
    """Normalize a value using log transformation (Bao-compatible)."""
    return (np.log(x + 1) - lo) / (hi - lo)


class StatExtractor:
    """Helper class for extracting and normalizing plan statistics (Bao-compatible)."""

    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        """Extract and normalize statistics from a node."""
        res = []
        for f, lo, hi in zip(self.__fields, self.__mins, self.__maxs):
            if f not in inp:
                res.append(0)
            else:
                res.append(norm(inp[f], lo, hi))
        return res


# Operator types from Bao
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg


def is_join(node):
    return node["Node Type"] in JOIN_TYPES


def is_scan(node):
    return node["Node Type"] in LEAF_TYPES


class TreeBuilder:
    """TreeBuilder class for compatibility with Bao's TreeFeaturizer."""
    
    def __init__(self, stats_extractor, relations):
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return np.concatenate((arr, self.__stats(node)))

    def __featurize_scan(self, node):
        assert is_scan(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return (np.concatenate((arr, self.__stats(node))),
                self.__relation_name(node))

    def plan_to_feature_tree(self, plan):
        children = plan["Plans"] if "Plans" in plan else []

        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])

        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            return self.__featurize_scan(plan)

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


def _attach_buf_data(tree):
    """Attach buffer data to tree (Bao-compatible function)."""
    if "Buffers" not in tree:
        return

    buffers = tree["Buffers"]

    def recurse(n):
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)
            return
        
        # it is a leaf
        n["Buffers"] = get_buffer_count_for_leaf(n, buffers)

    recurse(tree["Plan"])


def get_buffer_count_for_leaf(leaf, buffers):
    """Get buffer count for a leaf node (Bao-compatible function)."""
    total = 0
    if "Relation Name" in leaf:
        total += buffers.get(leaf["Relation Name"], 0)
    if "Index Name" in leaf:
        total += buffers.get(leaf["Index Name"], 0)
    return total


def get_all_relations(data):
    """Get all relations from plans (Bao-compatible function)."""
    all_rels = []
    
    def recurse(plan):
        if "Relation Name" in plan:
            yield plan["Relation Name"]
        if "Plans" in plan:
            for child in plan["Plans"]:
                yield from recurse(child)

    for plan in data:
        all_rels.extend(list(recurse(plan)))
        
    return set(all_rels)


def get_plan_stats(data):
    """Get plan statistics extractor (Bao-compatible function)."""
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
                recurse(child)

    for plan in data:
        recurse(plan, buffers=plan.get("Buffers", None))

    costs = np.array(costs)
    rows = np.array(rows)
    bufs = np.array(bufs) if bufs else np.array([])
    
    costs = np.log(costs + 1)
    rows = np.log(rows + 1)
    bufs = np.log(bufs + 1) if len(bufs) > 0 else np.array([])

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    bufs_min = np.min(bufs) if len(bufs) != 0 else 0
    bufs_max = np.max(bufs) if len(bufs) != 0 else 0

    if len(bufs) != 0:
        return StatExtractor(
            ["Buffers", "Total Cost", "Plan Rows"],
            [bufs_min, costs_min, rows_min],
            [bufs_max, costs_max, rows_max]
        )
    else:
        return StatExtractor(
            ["Total Cost", "Plan Rows"],
            [costs_min, rows_min],
            [costs_max, rows_max]
        )


class TreeFeaturizer:
    """
    Encoder for query plans based on Bao's TreeFeaturizer.
    Converts PostgreSQL EXPLAIN ANALYZE JSON plans into feature trees for neural network training.
    Compatible with Bao's model format.
    """

    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        """Fit the encoder on a set of plans (Bao-compatible interface)."""
        for t in trees:
            _attach_buf_data(t)
        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):
        """Transform plans into feature trees (Bao-compatible interface)."""
        for t in trees:
            _attach_buf_data(t)
        return [self.__tree_builder.plan_to_feature_tree(x) for x in trees]

    def num_operators(self):
        """Get the number of operator types."""
        return len(ALL_TYPES)


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
        "plan_json": plan_json if isinstance(plan_json, str) else json.dumps(plan_json)
    }

