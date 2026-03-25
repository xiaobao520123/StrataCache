from stratacache.backend.disk.engine.base import DiskBackendEngine
from stratacache.backend.base import BackendStats
from stratacache.core.artifact import ArtifactId, ArtifactMeta

import json
import threading
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

class FileEngine(DiskBackendEngine):
    def __init__(self, *, store_dir: str):
        self._name = "file"
        self._store_dir = Path(store_dir)
        self._lock = threading.RLock()
        self._bytes_used = 0
        self._items_count = 0
        
        # Create storage directory if it doesn't exist
        self._store_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        """Return the name of this engine."""
        return self._name
    
    def exists(self, artifact_id: ArtifactId) -> bool:
        """Check if an artifact exists in storage."""
        with self._lock:
            artifact_path = self._get_artifact_path(artifact_id)
            return artifact_path.exists()
    
    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        """Retrieve an artifact and its metadata from storage."""
        with self._lock:
            artifact_path = self._get_artifact_path(artifact_id)
            meta_path = self._get_meta_path(artifact_id)
            
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact {artifact_id} not found")
            
            # Read payload
            with open(artifact_path, 'rb') as f:
                payload = f.read()
            
            # Read metadata
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta_dict = json.load(f)
                meta = ArtifactMeta.from_json(meta_dict)
            else:
                meta = ArtifactMeta()
            
            return payload, meta
    
    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        """Store an artifact and its metadata."""
        with self._lock:
            artifact_path = self._get_artifact_path(artifact_id)
            meta_path = self._get_meta_path(artifact_id)
            
            # Ensure directory exists
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if artifact already exists to update bytes count
            bytes_to_add = len(payload)
            if artifact_path.exists():
                self._bytes_used -= artifact_path.stat().st_size
            else:
                self._items_count += 1
            
            # Write payload
            with open(artifact_path, 'wb') as f:
                f.write(payload)
            
            # Write metadata as JSON
            with open(meta_path, 'w') as f:
                json.dump(meta.to_json(), f)
            
            self._bytes_used += bytes_to_add
    
    def delete(self, artifact_id: ArtifactId) -> None:
        """Delete an artifact and its metadata from storage."""
        with self._lock:
            artifact_path = self._get_artifact_path(artifact_id)
            meta_path = self._get_meta_path(artifact_id)
            
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact {artifact_id} not found")
            
            # Update bytes count
            self._bytes_used -= artifact_path.stat().st_size
            self._items_count -= 1
            
            # Remove files
            artifact_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
    
    def stats(self) -> BackendStats:
        """Return statistics about the storage layer."""
        with self._lock:
            return BackendStats(
                items=self._items_count,
                bytes_used=self._bytes_used,
                bytes_capacity=None
            )
    
    def _get_artifact_path(self, artifact_id: ArtifactId) -> Path:
        """Get the file path for an artifact's payload."""
        return self._store_dir / f"{artifact_id.value}.bin"
    
    def _get_meta_path(self, artifact_id: ArtifactId) -> Path:
        """Get the file path for an artifact's metadata."""
        return self._store_dir / f"{artifact_id.value}.meta.json"
