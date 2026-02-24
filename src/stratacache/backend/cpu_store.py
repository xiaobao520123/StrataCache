from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

from stratacache.backend.base import BackendStats, MemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound


@dataclass(slots=True)
class _Entry:
    payload: bytes
    meta: ArtifactMeta

    @property
    def size(self) -> int:
        return len(self.payload)


class CpuMemoryLayer(MemoryLayer):
    """
    In-process memory store with best-effort LRU by bytes.

    This is intentionally simple for v0.1.
    """

    def __init__(self, *, capacity_bytes: Optional[int] = None, store_name: str = "cpu") -> None:
        self._name = store_name
        self._capacity_bytes = capacity_bytes
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, _Entry]" = OrderedDict()
        self._bytes_used = 0

    @property
    def name(self) -> str:
        return self._name

    def exists(self, artifact_id: ArtifactId) -> bool:
        k = str(artifact_id)
        with self._lock:
            return k in self._lru

    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        k = str(artifact_id)
        with self._lock:
            ent = self._lru.get(k)
            if ent is None:
                raise ArtifactNotFound(k)
            # Touch LRU
            self._lru.move_to_end(k, last=True)
            return ent.payload, ent.meta

    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        k = str(artifact_id)
        with self._lock:
            old = self._lru.get(k)
            if old is not None:
                self._bytes_used -= old.size
                self._lru.pop(k, None)

            ent = _Entry(payload=payload, meta=meta)
            self._lru[k] = ent
            self._bytes_used += ent.size
            self._lru.move_to_end(k, last=True)
            self._evict_if_needed()

    def delete(self, artifact_id: ArtifactId) -> None:
        k = str(artifact_id)
        with self._lock:
            ent = self._lru.pop(k, None)
            if ent is not None:
                self._bytes_used -= ent.size

    def stats(self) -> BackendStats:
        with self._lock:
            return BackendStats(
                items=len(self._lru),
                bytes_used=int(self._bytes_used),
                bytes_capacity=self._capacity_bytes,
            )

    def _evict_if_needed(self) -> None:
        if self._capacity_bytes is None:
            return
        while self._bytes_used > self._capacity_bytes and self._lru:
            _, ent = self._lru.popitem(last=False)  # least-recently used
            self._bytes_used -= ent.size


# Backward-compatible alias (v0.1)
CpuStore = CpuMemoryLayer

