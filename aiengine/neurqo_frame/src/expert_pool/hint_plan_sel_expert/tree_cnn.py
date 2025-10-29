import numpy as np
from typing import List, Union, Tuple, Optional, Callable, Any
import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()


class TreeCNN(nn.Module):
    """
    Neural network model for query plan latency prediction based on Tree's architecture.
    Uses tree convolutions to process query plan trees.
    """

    def __init__(self, in_channels: int):
        super(TreeCNN, self).__init__()
        self.in_channels = in_channels

        # Original Tree architecture with tree convolutions
        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        if USE_CUDA:
            self.cuda()

    def forward(self, x: List[Tuple]) -> torch.Tensor:
        """Forward pass through the network."""
        flat_trees = [_flatten(tree, self._features, self._left_child, self._right_child) for tree in x]
        flat_trees = _pad_and_combine(flat_trees)
        flat_trees = torch.Tensor(flat_trees)
        # flat trees is now batch x max tree nodes x channels
        flat_trees = flat_trees.transpose(1, 2)
        if USE_CUDA:
            flat_trees = flat_trees.cuda()
        indexes = [_tree_conv_indexes(tree, self._left_child, self._right_child) for tree in x]
        indexes = _pad_and_combine(indexes)
        indexes = torch.Tensor(indexes).long()
        if USE_CUDA:
            indexes = indexes.cuda()
        trees = (flat_trees, indexes)
        return self.tree_conv(trees)

    def _features(self, x: Tuple) -> np.ndarray:
        """Extract features from tree node."""
        return x[0]

    def _left_child(self, x: Tuple) -> Optional[Tuple]:
        """Get left child of tree node."""
        if len(x) != 3:
            return None
        return x[1]

    def _right_child(self, x: Tuple) -> Optional[Tuple]:
        """Get right child of tree node."""
        if len(x) != 3:
            return None
        return x[2]


class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        zero_vec = zero_vec.to(results.device)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)


class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.activation(x[0]), x[1])


class TreeLayerNorm(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)


def _is_leaf(x: Tuple, left_child: Callable, right_child: Callable) -> bool:
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None

    if has_left != has_right:
        raise Exception(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left


def _flatten(root: Tuple, transformer: Callable, left_child: Callable, right_child: Callable) -> np.ndarray:
    """ turns a tree into a flattened vector, preorder """

    if not callable(transformer):
        raise Exception(
            "Transformer must be a function mapping a tree node to a vector"
        )

    if not callable(left_child) or not callable(right_child):
        raise Exception(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    accum: List[np.ndarray] = []

    def recurse(x: Tuple):
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return

        accum.append(transformer(x))
        recurse(left_child(x))
        recurse(right_child(x))

    recurse(root)

    try:
        accum = [np.zeros(accum[0].shape)] + accum
    except:
        raise Exception(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )

    return np.array(accum)


def _preorder_indexes(root: Tuple, left_child: Callable, right_child: Callable, idx: int = 1) -> Union[int, Tuple]:
    """ transforms a tree into a tree of preorder indexes """

    if not callable(left_child) or not callable(right_child):
        raise Exception(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )

    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree: Union[int, Tuple]) -> int:
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree

    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx + 1)

    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)


def _tree_conv_indexes(root: Tuple, left_child: Callable, right_child: Callable) -> np.ndarray:
    """
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """

    if not callable(left_child) or not callable(right_child):
        raise Exception(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root: Union[int, Tuple]) -> List[List[int]]:
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            yield [my_id, left_id, right_id]

            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]

    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)


def _pad_and_combine(x: List[np.ndarray]) -> np.ndarray:
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise Exception(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )

    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim

    max_first_dim = max(arr.shape[0] for arr in x)

    vecs: List[np.ndarray] = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)


class DynamicPooling(nn.Module):
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.max(x[0], dim=2).values
