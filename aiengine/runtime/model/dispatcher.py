"""Model registry: name -> box class.

Same self-registration pattern as ``NullFillBuilder``: adding a model to
the zoo is one class + one decorator, no dispatcher edits. The registry
key is what ``TrainingSpec.model_name`` carries over the wire.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from .base import Model, ModelSpec


class ModelDispatcher:
    _registry: Dict[str, Type[Model]] = {}
    _unavailable: Dict[str, str] = {}  # name -> reason (missing optional dep)

    @classmethod
    def register(cls, name: str) -> Callable[[Type[Model]], Type[Model]]:
        def decorator(model_cls: Type[Model]) -> Type[Model]:
            if name in cls._registry:
                raise ValueError(f"model already registered: {name!r}")
            cls._unavailable.pop(name, None)
            model_cls.name = name
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def register_unavailable(cls, name: str, reason: str) -> None:
        """Mark a known model as unbuildable (optional dependency missing),
        so dispatching it fails with the actionable reason instead of
        'unknown model'."""
        if name not in cls._registry:
            cls._unavailable[name] = reason

    @classmethod
    def build(
        cls,
        name: str,
        spec: ModelSpec,
        params: Optional[Dict[str, Any]] = None,
    ) -> Model:
        model_cls = cls._registry.get(name)
        if model_cls is None:
            reason = cls._unavailable.get(name)
            if reason is not None:
                raise RuntimeError(f"model {name!r} is unavailable: {reason}")
            raise ValueError(
                f"unknown model {name!r}; registered: {sorted(cls._registry)}"
            )
        return model_cls(spec=spec, params=params)

    @classmethod
    def registered(cls) -> Dict[str, Type[Model]]:
        return dict(cls._registry)
