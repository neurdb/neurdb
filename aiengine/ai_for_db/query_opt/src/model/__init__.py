"""Workload-agnostic encoders and hierarchical policy models."""

from typing import Any

__all__ = ["HACNetwork", "HierarchicalActorCritic"]


def __getattr__(name: str) -> Any:
    """Load the PyTorch policy only when checkpoint inference needs it."""
    if name in __all__:
        from .policy.hierarchical_actor_critic import HACNetwork

        return HACNetwork
    raise AttributeError(name)
