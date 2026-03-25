from stratacache.backend.base import MemoryLayer, BackendStats
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.backend.disk.engine.base import DiskBackendEngine
from stratacache.backend.disk.engine.file.engine import FileEngine
from stratacache.backend.disk.engine.lsm.engine import LSMEngine

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

class DiskMemoryLayer(MemoryLayer):
    def __init__(
        self,
        *,
        store_name: str = "disk",
        store_dir: str = "./disk_store",
        engine: str = "lsm",
    ):
        self._name = store_name
        self._store_dir = Path(store_dir)
        self._engine = self.init_engine(engine, store_dir)

    @property
    def name(self) -> str:
        """Return the name of this memory layer."""
        return self._name

    def init_engine(self, engine_name: str, store_dir: str) -> DiskBackendEngine:
        """Initialize the appropriate disk backend engine based on the configuration."""
        if engine_name == "file":
            return FileEngine(store_dir=store_dir)
        elif engine_name == "lsm":
            return LSMEngine(store_dir=store_dir)
        else:
            raise ValueError(f"Unsupported disk backend engine: {engine_name}")

    def exists(self, artifact_id: ArtifactId) -> bool:
        """Check if an artifact exists in storage."""
        return self._engine.exists(artifact_id)

    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        """Retrieve an artifact and its metadata from storage."""
        return self._engine.get(artifact_id)

    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        """Store an artifact and its metadata."""
        self._engine.put(artifact_id, payload, meta)

    def delete(self, artifact_id: ArtifactId) -> None:
        """Delete an artifact and its metadata from storage."""
        self._engine.delete(artifact_id)

    def stats(self) -> BackendStats:
        """Return statistics about the storage layer."""
        return self._engine.stats()


DiskStore = DiskMemoryLayer
