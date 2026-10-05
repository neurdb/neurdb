"""Unified optimization actions and their execution contracts."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .actions import ActionProfile

__all__ = ["ActionProfile"]


def __getattr__(name: str) -> Any:
    """Load public action types lazily to keep compatibility adapters acyclic."""
    if name == "ActionProfile":
        from .actions import ActionProfile

        return ActionProfile
    raise AttributeError(name)
