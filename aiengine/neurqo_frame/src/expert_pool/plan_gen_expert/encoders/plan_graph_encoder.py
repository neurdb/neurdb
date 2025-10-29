import time
from typing import Optional, Tuple, Any, Union, List, Dict

import numpy as np
import torch
from common import workload


class Featurizer:

    def __call__(self, node):
        """Node -> np.ndarray."""
        raise NotImplementedError

    def FeaturizeLeaf(self, node):
        """Featurizes a leaf Node."""
        raise NotImplementedError

    def Merge(self, node, left_vec, right_vec):
        """Featurizes a Node by merging the feature vectors of LHS/RHS."""
        raise NotImplementedError

    def Fit(self, nodes):
        """Computes normalization statistics; no-op for stateless."""
        return

    def PerturbQueryFeatures(self, query_feat, distribution):
        """Randomly perturbs a query feature vec returned by __call__()."""
        return query_feat

    # def WithWorkloadInfo(self, workload_info):
    #     self.workload_info = workload_info
    #     return self


class SimPlanFeaturizer(Featurizer):
    """Implements the plan featurizer.

        plan node -> [ multi-hot of tables on LHS ] [ same for RHS ]
    """

    def __init__(self, workload_info):
        self.workload_info = workload_info

    def __call__(self, node):
        vec = np.zeros(self.dims, dtype=np.float32)

        # Tables on LHS.
        for rel_id in node.children[0].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx] = 1.0

        # Tables on RHS.
        for rel_id in node.children[1].leaf_ids():
            idx = np.where(self.workload_info.rel_ids == rel_id)[0][0]
            vec[idx + len(self.workload_info.rel_ids)] = 1.0

        return vec

    @property
    def dims(self):
        return len(self.workload_info.rel_ids) * 2


class TreeNodeFeaturizer(Featurizer):
    """Featurizes a single Node.

    Feature vector:
       [ one-hot for operator ] [ multi-hot for all relations under this node ]

    Width: |all_ops| + |rel_ids|.
    """

    def __init__(self, workload_info: workload.WorkloadInfo):
        self.workload_info = workload_info
        self.ops = workload_info.all_ops
        self.one_ops = np.eye(self.ops.shape[0], dtype=np.float32)
        self.rel_ids = workload_info.rel_ids
        assert not workload_info.HasPhysicalOps(), 'Physical ops found; use a ' \
                                                   'featurizer that supports them (PhysicalTreeNodeFeaturizer).'

    def __call__(self, node: workload.Node):
        num_ops = len(self.ops)
        vec = np.zeros(num_ops + len(self.rel_ids), dtype=np.float32)
        # Node type.
        vec[:num_ops] = self.one_ops[np.where(self.ops == node.node_type)[0][0]]
        # Joined tables: [table: 1].
        joined = node.leaf_ids()
        for rel_id in joined:
            idx = np.where(self.rel_ids == rel_id)[0][0]
            vec[idx + num_ops] = 1.0
        assert vec[num_ops:].sum() == len(joined)
        return vec

    def FeaturizeLeaf(self, node: workload.Node):
        assert node.IsScan()
        vec = np.zeros(len(self.ops) + len(self.rel_ids), dtype=np.float32)
        rel_id = node.get_table_id()
        rel_idx = np.where(self.rel_ids == rel_id)[0][0]
        vec[len(self.ops) + rel_idx] = 1.0
        return vec

    def Merge(self, node: workload.Node, left_vec, right_vec):
        assert node.IsJoin()
        len_join_enc = len(self.ops)
        # The relations under 'node' and their scan types.  Merging <=> summing.
        vec = left_vec + right_vec
        # Make sure the first part is correct.
        vec[:len_join_enc] = self.one_ops[np.where(
            self.ops == node.node_type)[0][0]]
        return vec


class PhysicalTreeNodeFeaturizer(Featurizer):
    """Featurizes a single Node with support for physical operators.

    Feature vector:
       [ one-hot for join operator ] concat
       [ multi-hot for all relations under this node ]

    Width: |join_ops| + |rel_ids| * |scan_ops|.
    """

    def __init__(self, workload_info: workload.WorkloadInfo):
        # super().__init__(workload_info)
        self.workload_info = workload_info
        self.join_ops = workload_info.join_types
        self.scan_ops = workload_info.scan_types
        self.rel_ids = workload_info.rel_ids
        self.join_one_hot = np.eye(len(self.join_ops), dtype=np.float32)

    def __call__(self, node):
        # Join op of this node.
        if node.IsJoin():
            join_encoding = self.join_one_hot[np.where(
                self.join_ops == node.node_type)[0][0]]
        else:
            join_encoding = np.zeros(len(self.join_ops), dtype=np.float32)
        # For each table: [ one-hot of scan ops ].  Concat across tables.
        scan_encoding = np.zeros(len(self.scan_ops) * len(self.rel_ids),
                                 dtype=np.float32)
        joined = node.CopyLeaves()
        for rel_node in joined:
            rel_id = rel_node.get_table_id()
            rel_idx = np.where(self.rel_ids == rel_id)[0][0]
            scan_operator_idx = np.where(
                self.scan_ops == rel_node.node_type)[0][0]
            idx = rel_idx * len(self.scan_ops) + scan_operator_idx
            scan_encoding[idx] = 1.0
        # Concatenate to create final node encoding.
        vec = np.concatenate((join_encoding, scan_encoding))
        return vec

    def FeaturizeLeaf(self, node):
        assert node.IsScan()
        vec = np.zeros(len(self.join_ops) +
                       len(self.scan_ops) * len(self.rel_ids),
                       dtype=np.float32)  # here is [table num * scan num + join number]
        rel_id = node.get_table_id()
        rel_idx = np.where(self.rel_ids == rel_id)[0][0]
        scan_operator_idx = np.where(self.scan_ops == node.node_type)[0][0]
        idx = rel_idx * len(self.scan_ops) + scan_operator_idx
        vec[len(self.join_ops) + idx] = 1.0
        return vec

    def Merge(self, node, left_vec, right_vec):
        assert node.IsJoin()
        len_join_enc = len(self.join_ops)
        # The relations under 'node' and their scan types.  Merging <=> summing.
        vec = left_vec + right_vec
        # Make sure the first part is correct.
        vec[:len_join_enc] = self.join_one_hot[np.where(
            self.join_ops == node.node_type)[0][0]]
        return vec


class PreOrderSequenceFeaturizer(Featurizer):

    def __init__(self, workload_info):
        self.workload_info = workload_info

        # Union of all relation names and operator types.
        self.vocab = np.concatenate(
            (workload_info.all_ops, workload_info.rel_names))

        print('PreOrderSequenceFeaturizer vocab', self.vocab)

    def pad(self):
        return len(self.vocab)

    def _pre_order(self, parent, n, vecs):
        """Each node yields up to 2 tokens: <op type> <rel name (if scan)>."""
        vecs.append(np.where(self.vocab == n.node_type)[0][0])
        if len(n.children) == 0:
            name = n.table_name
            vecs.append(np.where(self.vocab == name)[0][0])
        else:
            for c in n.children:
                self._pre_order(n, c, vecs)

    def __call__(self, node):
        vecs = []
        self._pre_order(None, node, vecs)
        return np.asarray(vecs).astype(np.int64, copy=False)


class ParentPositionFeaturizer(Featurizer):
    """Node -> parent ID, where IDs are assigned DFS-order."""

    def __init__(self, workload_info):
        self.pad_idx = len(workload_info.rel_names)
        self.pad_idx = 50  # FIXME

    def pad(self):
        return self.pad_idx

    def _walk(self, parent, n, vecs, parent_id, curr_id):
        vecs.append(parent_id)
        if len(n.children) == 0:
            # [Scan, TableName] corresponds to the same parent.
            vecs.append(parent_id)
        for c in n.children:
            self._walk(n, c, vecs, parent_id=curr_id, curr_id=curr_id + 1)

    def __call__(self, node):
        vecs = []
        self._walk(None, node, vecs, parent_id=-1, curr_id=0)
        vecs = np.asarray(vecs).astype(np.int64, copy=False)
        vecs += 1  # Shift by one since we have -1 for root.
        return np.arange(len(vecs))  # FIXME


def make_and_featurize_trees(node_featurizer, trees: List[workload.Node]):
    t1 = time.time()
    indexes = torch.from_numpy(_batch([_make_indexes(x) for x in trees])).long()
    trees = torch.from_numpy(
        _batch([_featurize_tree(x, node_featurizer) for x in trees
                ])).transpose(1, 2)
    # print('took {:.1f}s'.format(time.time() - t1))
    return trees, indexes


def _featurize_tree(curr_node: workload.Node, node_featurizer):
    def _bottom_up(curr):
        """Calls node_featurizer on each node exactly once, bottom-up."""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            vec = node_featurizer.FeaturizeLeaf(curr)
            curr.__node_feature_vec = vec
            return vec

        left_vec = _bottom_up(curr.children[0])
        right_vec = _bottom_up(curr.children[1])
        vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)
    vecs = []
    workload.NodeOps.map_node(curr_node,
                              lambda node: vecs.append(node.__node_feature_vec))

    # Add a zero-vector at index 0.
    ret = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    ret[1:] = vecs
    return ret


def _batch(data):
    lens = [vec.shape[0] for vec in data]
    if len(set(lens)) == 1:
        # Common path.
        return np.asarray(data)
    xs = np.zeros((len(data), np.max(lens), data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs


def _make_indexes(root: workload.Node):
    # Join(A, B) --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C) --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    preorder_ids, _ = _make_preorder_ids_tree(root)
    vecs = []
    _walk(preorder_ids, vecs)
    # Continuing with the Join(A,B) example:
    # Preorder traversal _walk() produces
    #   [1, 2, 3]
    #   [2, 0, 0]
    #   [3, 0, 0]
    # which would be reshaped into
    #   array([[1],
    #          [2],
    #          [3],
    #          [2],
    #          [0],
    #          [0],
    #    ...,
    #          [0]])
    vecs = np.asarray(vecs).reshape(-1, 1)
    return vecs


def _make_preorder_ids_tree(curr: workload.Node, root_index=1):
    """Returns a tuple containing a tree of preorder positional IDs.

    Returns (tree structure, largest id under me).  The tree structure itself
    (the first slot) is a 3-tuple:

    If curr is a leaf:
      tree structure is (my id, 0, 0) (note that valid IDs start with 1)
    Else:
      tree structure is
        (my id, tree structure for LHS, tree structure for RHS).

    This function traverses each node exactly once (i.e., O(n) time complexity).
    """
    if not curr.children:
        return (root_index, 0, 0), root_index

    # Under every Bitmap Heap Scan is a Bitmap Index Scan, these do not need to be
    # considered seperately -> directly act as if the Bitmap Heap Scan was the leaf node
    #
    if curr.node_type == 'Bitmap Heap Scan':
        return (root_index, 0, 0), root_index

    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0],
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1],
                                              root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


def _walk(curr, vecs):
    if curr[1] == 0:
        # curr is a leaf.
        vecs.append(curr)

    else:
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)
        _walk(curr[2], vecs)
