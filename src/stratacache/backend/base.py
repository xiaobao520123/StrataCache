from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from stratacache.core.artifact import ArtifactId, ArtifactMeta


@dataclass(frozen=True, slots=True)
class BackendStats:
    items: int
    bytes_used: int
    bytes_capacity: Optional[int] = None


class MemoryLayer(ABC):
    """
    Minimal memory layer contract.

    v0.1 uses bytes payloads and JSON-like metadata (encoded by the caller).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def exists(self, artifact_id: ArtifactId) -> bool: ...

    @abstractmethod
    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]: ...

    @abstractmethod
    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None: ...

    @abstractmethod
    def delete(self, artifact_id: ArtifactId) -> None: ...

    @abstractmethod
    def stats(self) -> BackendStats: ...


# Backward-compatible aliases (v0.1)
StorageBackend = MemoryLayer
LayerStats = BackendStats

