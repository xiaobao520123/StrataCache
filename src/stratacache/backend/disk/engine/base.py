from stratacache.backend.base import BackendStats
from stratacache.core.artifact import ArtifactId, ArtifactMeta

from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod

class DiskBackendEngine(ABC):
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        
    @abstractmethod
    def exists(self, artifact_id: ArtifactId) -> bool:
        """Check if an artifact exists in storage."""
        pass
    
    @abstractmethod
    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        """Retrieve an artifact and its metadata from storage."""
        pass
    
    @abstractmethod
    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        """Store an artifact and its metadata."""
        pass
    
    @abstractmethod
    def delete(self, artifact_id: ArtifactId) -> None:
        """Delete an artifact and its metadata from storage."""
        pass
    
    @abstractmethod
    def stats(self) -> BackendStats:
        """Return statistics about the backend storage."""
        pass