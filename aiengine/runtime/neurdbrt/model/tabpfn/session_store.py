"""In-process cache of fitted in-context models, keyed by ``model_id``.

A single ``PREDICT ... TRAIN tabpfn`` runs as two phases against the AI engine:
the context (train) phase fits a :class:`~.stateful.StatefulTabPFN` and the
predict (inference) phase reuses it. The DB engine reconnects between the two
phases, so the fitted context must outlive the train websocket session. We keep
it in a module-level dict (the AI server runs as a single uvicorn worker), keyed
by the synthetic ``model_id`` the engine returns from the train phase and then
passes back in the inference task.

For distributed inference the DB engine broadcasts the context (train) phase
to every registered engine; each process fits the same context and stores it
here under its own process-local model id, which the engine passes back in
that worker's inference task.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from .stateful import StatefulTabPFN

_LOCK = threading.Lock()
_CONTEXTS: Dict[int, StatefulTabPFN] = {}
_NEXT_ID = 1


def new_model_id() -> int:
    """Allocate a process-unique model id for a freshly fitted context."""
    global _NEXT_ID
    with _LOCK:
        mid = _NEXT_ID
        _NEXT_ID += 1
        return mid


def put(model_id: int, model: StatefulTabPFN) -> None:
    with _LOCK:
        _CONTEXTS[int(model_id)] = model


def get(model_id: int) -> Optional[StatefulTabPFN]:
    with _LOCK:
        return _CONTEXTS.get(int(model_id))


def remove(model_id: int) -> None:
    with _LOCK:
        _CONTEXTS.pop(int(model_id), None)


def contains(model_id: int) -> bool:
    with _LOCK:
        return int(model_id) in _CONTEXTS
