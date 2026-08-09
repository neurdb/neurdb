"""TorchMiddleware: EncodedBatch -> ModelInput, the torch family's packager.

All NON-learnable conversion lives here, outside the module graph: the
learnable lookups (``TokenEmbedding``, future ``TimestampEmbedding``) stay
inside the model, everything that merely rearranges data runs once per
batch in the driver. Steps plug in per box requirements — tokenization is
always on (every torch model needs it); edges join for RELATIONAL boxes;
time channels are a future step.

``ModelInput`` is the data package torch models consume: numpy until
``.to(device)`` materializes torch tensors (the single torch touchpoint,
so this module imports without torch and the packaging logic is testable
torch-less). Following PyG's ``Data``/``HeteroData``, the package carries
``y`` — convention: ``y`` is for drivers and losses; architectures'
``forward`` must never read it. At inference the middleware is built with
``require_target=False`` and ``y`` is None, so any model that cheated
breaks loudly the first time it serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np
from pipeline.input import EncodedBatch

from ..base import Model, ModelKind, ModelSpec
from ..tokenizer import FeatureTokenizer


@dataclass(frozen=True)
class ModelInput:
    """The torch family's data package. Arrays are numpy until ``.to``.

    ``tokens``/``values`` are per-table ``(B_t, N_t)`` global token ids and
    cell values (see ``model.tokenizer``); ``edges`` maps relationship name
    to a ``(2, M)`` index pair when the box asked for graph structure;
    ``target_table`` names the prediction anchor.
    """

    tokens: Mapping[str, Any]
    values: Mapping[str, Any]
    target_table: str
    edges: Optional[Mapping[str, Any]] = None
    y: Optional[Any] = None

    def to(self, device: Any) -> "ModelInput":
        """Materialize torch tensors on ``device`` — the one torch touchpoint."""
        import torch  # deliberate local import: the class must load torch-less

        def conv(a: Any) -> Any:
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        return ModelInput(
            tokens={t: conv(a) for t, a in self.tokens.items()},
            values={t: conv(a) for t, a in self.values.items()},
            target_table=self.target_table,
            edges=(
                None
                if self.edges is None
                else {r: conv(a) for r, a in self.edges.items()}
            ),
            y=None if self.y is None else conv(self.y),
        )


class TorchStep(Protocol):
    """Pluggable packaging step: contribute a slice of the package."""

    def contribute(
        self, encoded: EncodedBatch, spec: ModelSpec, package: Dict[str, Any]
    ) -> None: ...


class TokenizeStep:
    """Always installed: features -> per-table global tokens/values."""

    def __init__(self, tokenizer: FeatureTokenizer) -> None:
        self._tokenizer = tokenizer

    def contribute(self, encoded, spec, package) -> None:
        per_table = self._tokenizer.transform(encoded.features)
        package["tokens"] = {t: tv[0] for t, tv in per_table.items()}
        package["values"] = {t: tv[1] for t, tv in per_table.items()}


class EdgesStep:
    """RELATIONAL boxes: RelationGraph -> {rel: (2, M) int64}."""

    def contribute(self, encoded, spec, package) -> None:
        package["edges"] = {
            name: np.stack([src, tgt])
            for name, (src, tgt) in encoded.graph.edges.items()
        }


class TorchMiddleware:
    def __init__(
        self,
        spec: ModelSpec,
        steps: Tuple[TorchStep, ...] = (),
        require_target: bool = True,
    ) -> None:
        self._spec = spec
        self._steps: Tuple[TorchStep, ...] = (
            TokenizeStep(FeatureTokenizer(spec)),
        ) + steps
        self._require_target = require_target

    @classmethod
    def for_model(cls, model: Model, require_target: bool = True) -> "TorchMiddleware":
        """Select steps from the box's declared requirements."""
        steps: Tuple[TorchStep, ...] = (
            (EdgesStep(),) if model.kind is ModelKind.RELATIONAL else ()
        )
        return cls(model.spec, steps=steps, require_target=require_target)

    def __call__(self, encoded: EncodedBatch) -> ModelInput:
        package: Dict[str, Any] = {}
        for step in self._steps:
            step.contribute(encoded, self._spec, package)

        y = encoded.features.features.get(self._spec.target)
        if y is None and self._require_target:
            raise KeyError(
                f"target column {self._spec.target} is missing from the "
                "batch; training requires it"
            )
        return ModelInput(
            tokens=package["tokens"],
            values=package["values"],
            target_table=self._spec.target[0],
            edges=package.get("edges"),
            y=y,
        )
