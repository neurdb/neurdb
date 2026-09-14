"""
Hint Plan Selection Expert Package - TreeCNN-based query optimizer expert.

This expert selects the best query plan using hint-based optimization.
Contains:
- model.py: Main expert implementation (Bao-compatible, renamed from tree_based_expert.py)
- featurize.py: Plan encoder (TreeFeaturizer, Bao-compatible)
- tree_cnn.py: Neural network model (TreeCNN)
"""

from expert_pool.hint_plan_sel_expert.model import TreeCNNConfig, TreeCNNRegExpert

__all__ = ["TreeCNNRegExpert", "TreeCNNConfig"]
