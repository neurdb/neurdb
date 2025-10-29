import numpy as np
import torch
from typing import List, Tuple, Dict
from expert_pool.join_order_expert.tools.normalize import LatencyNormalizer


class TreeBuilder:
    """
    Turns a PG plan JSON into a feature tree suitable for a model.
    - Holds config, type lists, value encoding, and all helpers.
    - No global constants; JOIN/SCAN types live on the instance.
    """

    def __init__(self, id2aliasname: Dict, aliasname2id: Dict, input_size: int, hidden_size: int,
                 latency_normalizer: LatencyNormalizer = None):
        self.JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
        self.LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
        self.ALL_TYPES = self.JOIN_TYPES + self.LEAF_TYPES

        self.id2aliasname = id2aliasname
        self.aliasname2id = aliasname2id

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.latency_normalizer = latency_normalizer

    # ---- utility formerly-global helpers ----
    def zero_hc(self, input_dim: int = 1):
        """Replaces global zero_hc; returns (h0, c0) on correct device/dtype."""
        hs = self.hidden_size
        return torch.zeros(input_dim, hs), torch.zeros(input_dim, hs)

    def _get_plan_stats(self, data: dict) -> List[float]:
        return [
            self.latency_normalizer.encode(data["Total Cost"]),
            self.latency_normalizer.encode(data["Plan Rows"]),
        ]

    # ---- node-type predicates ----
    def _is_join(self, node: dict) -> bool:
        return node.get("Node Type") in self.JOIN_TYPES

    def _is_scan(self, node: dict) -> bool:
        return node.get("Node Type") in self.LEAF_TYPES

    # ---- internal extractors ----
    def _alias_name(self, node: dict) -> np.ndarray:
        if "Alias" in node:
            return np.asarray([self.aliasname2id[node["Alias"]]])

        if node.get("Node Type") == "Bitmap Index Scan":
            name_key = "Index Cond"
            if name_key not in node:
                raise Exception("Bitmap operator missing Index Cond")
            for rel in self.aliasname2id:
                if rel + "." in node[name_key]:
                    # original code had an unreachable -1 then a valid return;
                    # we keep the valid path.
                    return np.asarray([self.aliasname2id[rel]])

        raise Exception("Cannot extract Alias from node: " + str(node))

    def _featurize_join(self, node: dict) -> torch.Tensor:
        assert self._is_join(node)
        arr = np.zeros(len(self.ALL_TYPES))
        arr[self.ALL_TYPES.index(node["Node Type"])] = 1.0
        feature = np.concatenate((arr, self._get_plan_stats(node)))
        return torch.tensor(feature, dtype=torch.float32).reshape(-1, self.input_size)

    def _featurize_scan(self, node: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._is_scan(node)
        arr = np.zeros(len(self.ALL_TYPES))
        arr[self.ALL_TYPES.index(node["Node Type"])] = 1.0
        feature = np.concatenate((arr, self._get_plan_stats(node)))
        feat_t = torch.tensor(feature, dtype=torch.float32).reshape(-1, self.input_size)
        alias_idx_np = self._alias_name(node)
        alias_t = torch.tensor(alias_idx_np, dtype=torch.long)
        return feat_t, alias_t

    # ---- public API ----
    def plan_to_feature_tree(self, plan: dict):
        """
        Convert a PG plan dict into a feature tree:
          - join node => (feat_tensor, left_subtree, right_subtree)
          - scan node => (feat_tensor, alias_tensor)
          - transparent single-child nodes are collapsed
        """
        node = plan.get("Plan", plan)
        children = node.get("Plan", node.get("Plans", []))

        if len(children) == 1:
            child_value = self.plan_to_feature_tree(children[0])
            if "Alias" in node and node.get("Node Type") == "Bitmap Heap Scan":
                alias_idx_np = np.asarray([self.aliasname2id[node["Alias"]]])
                if isinstance(child_value[1], tuple):
                    raise Exception("Unexpected child tuple for transparent node: " + str(node))
                return (child_value[0], torch.tensor(alias_idx_np, dtype=torch.long))
            return child_value

        if self._is_join(node):
            assert len(children) == 2, "Join node must have exactly two children"
            my_vec = self._featurize_join(node)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            return (my_vec, left, right)

        if self._is_scan(node):
            assert not children, "Scan node should not have children"
            return self._featurize_scan(node)

        raise Exception("Node wasn't transparent, a join, or a scan: " + str(node))
