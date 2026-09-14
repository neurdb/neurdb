"""Hierarchical policy network and action-head definitions."""

from .hierarchical_actor_critic import HACNetwork

HierarchicalActorCritic = HACNetwork

__all__ = ["HACNetwork", "HierarchicalActorCritic"]
