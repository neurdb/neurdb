"""In-memory dataset management with pinning and LRU eviction.

The scheduler owns a single ``DatasetManager``. Each job acquires its
dataset by *cache key* — ``task.metadata["dataset_key"]`` when the DB
engine provides one, else the ``task_id``. Sharing a key across tasks is
the reuse contract: a second task on the same key trains straight from the
warm cache and never touches the wire (its ``DataSource`` is not read).

Lifecycle rules the manager enforces:

* **Fill once** — the first acquirer drains the one-shot engine stream
  into the entry; concurrent acquirers of the same key wait for that fill
  instead of re-streaming.
* **Pin while running** — an acquired entry cannot be evicted until every
  job using it releases it. A mid-training eviction would be fatal: the
  engine's stream cannot be replayed.
* **Retain, then LRU** — released entries stay warm for future reuse and
  are evicted least-recently-used only when a new fill needs room under
  ``budget_bytes``. The budget is soft: if evicting every unpinned entry
  still cannot make room (all data belongs to running jobs), the fill
  proceeds over budget with a warning rather than failing the job.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from data.batch import DataBatch
from data.io import DataSource

logger = logging.getLogger(__name__)


def _batch_nbytes(batch: DataBatch) -> int:
    return sum(rb.nbytes for rb in batch.tables.values())


class CachedDataset:
    """One cached dataset. Satisfies the ``DataCache`` protocol: ``fill``
    consumes the one-shot stream exactly once; iteration is re-usable
    thereafter and yields frames in arrival order."""

    def __init__(self) -> None:
        self._batches: List[DataBatch] = []
        self.nbytes = 0

    def fill(self, source: DataSource) -> None:
        for batch in source:
            self._batches.append(batch)
            self.nbytes += _batch_nbytes(batch)

    def __iter__(self) -> Iterator[DataBatch]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


@dataclass
class _Entry:
    dataset: CachedDataset = field(default_factory=CachedDataset)
    ready: threading.Event = field(default_factory=threading.Event)
    fill_error: Optional[BaseException] = None
    pins: int = 0
    last_used: int = 0


class DatasetManager:
    """Keyed dataset store with fill-once, pinning, and soft-budget LRU."""

    def __init__(self, budget_bytes: int) -> None:
        self._budget = budget_bytes
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._tick = 0

    def acquire(self, key: str, source: DataSource) -> CachedDataset:
        """Return the READY dataset for ``key``, pinned for the caller.

        First acquirer fills from ``source``; concurrent acquirers block
        until that fill completes. Every successful acquire must be paired
        with exactly one ``release``.
        """
        with self._lock:
            entry = self._entries.get(key)
            filler = entry is None
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
            entry.pins += 1
            entry.last_used = self._next_tick()

        if filler:
            try:
                entry.dataset.fill(source)
                with self._lock:
                    self._evict_for(entry)
            except BaseException as exc:
                with self._lock:
                    entry.fill_error = exc
                    entry.pins -= 1
                    del self._entries[key]
                entry.ready.set()
                raise
            entry.ready.set()
        else:
            entry.ready.wait()
            if entry.fill_error is not None:
                with self._lock:
                    entry.pins -= 1
                raise RuntimeError(
                    f"cached dataset {key!r} failed to fill"
                ) from entry.fill_error

        return entry.dataset

    def release(self, key: str) -> None:
        """Unpin; the entry stays warm until LRU eviction needs the room."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.pins = max(0, entry.pins - 1)
            entry.last_used = self._next_tick()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(
                e.dataset.nbytes for e in self._entries.values() if e.ready.is_set()
            )

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def _evict_for(self, keep: _Entry) -> None:
        """Evict LRU unpinned entries until under budget. Caller holds lock."""

        def total() -> int:
            return sum(
                e.dataset.nbytes
                for e in self._entries.values()
                if e.ready.is_set() or e is keep
            )

        while total() > self._budget:
            victims = [
                (entry.last_used, key)
                for key, entry in self._entries.items()
                if entry.pins == 0 and entry.ready.is_set() and entry is not keep
            ]
            if not victims:
                logger.warning(
                    "dataset storage over budget (%d > %d bytes) but every entry "
                    "is pinned by a running job; proceeding over budget",
                    total(),
                    self._budget,
                )
                return
            _, victim_key = min(victims)
            evicted = self._entries.pop(victim_key)
            logger.info(
                "evicting cached dataset %r (%d bytes, LRU)",
                victim_key,
                evicted.dataset.nbytes,
            )
