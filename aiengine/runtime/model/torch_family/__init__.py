"""Torch model family: GRADIENT boxes, single-table and (future) relational.

The optional-dependency guard lives HERE, in the family package itself:
the torch-free middleware imports unconditionally (its packaging logic is
testable without torch — ``.to(device)`` is the one torch touchpoint);
the torch-dependent modules import under a try, and when torch is missing
the family's model names register as unavailable so dispatching them fails
with an actionable message instead of "unknown model".

Relational (multi-table, GNN) models belong to this family too — they
will live here as additional modules guarded the same way on their extra
dependency (torch_geometric / pyg), consuming ``ModelInput.edges`` from
the same middleware.
"""

from ..dispatcher import ModelDispatcher
from .middleware import (
    EdgesStep,
    ModelInput,
    TokenizeStep,
    TorchMiddleware,
    TorchStep,
)

# Every model name in this family — kept here (outside the torch-dependent
# modules) because listing them must not require importing torch.
_TORCH_MODELS = ("mlp",)

try:
    from .base import TorchModel
    from .embedding import TokenEmbedding, TorchModule
    from .mlp import MLPModel
except ImportError:
    TorchModel = None  # type: ignore[assignment]
    TokenEmbedding = None  # type: ignore[assignment]
    TorchModule = None  # type: ignore[assignment]
    MLPModel = None  # type: ignore[assignment]
    for _name in _TORCH_MODELS:
        ModelDispatcher.register_unavailable(
            _name,
            "requires torch; install the runtime with the 'cpu' or 'cuda' "
            "extra (e.g. `uv sync --extra cpu`)",
        )

__all__ = [
    "ModelInput",
    "TorchMiddleware",
    "TorchStep",
    "TokenizeStep",
    "EdgesStep",
    "TorchModel",
    "TorchModule",
    "TokenEmbedding",
    "MLPModel",
]
