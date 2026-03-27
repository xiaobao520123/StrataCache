from stratacache.backend.disk.engine.base import DiskBackendEngine
from stratacache.backend.base import BackendStats
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound
from dataclasses import dataclass, field

from pathlib import Path
from typing import Tuple
from stratacache.backend.disk.engine.lsm.memtable import MemTable
import logging
from struct import pack, unpack

import mmap

logger = logging.getLogger(__name__)

MAGIC = b"SLSM"  # Magic header for LSM files
from hashlib import sha256

@dataclass
class LSMBlockValue:
    file_id: int = 0
    offset: int = 0
    meta: ArtifactMeta = field(default_factory=ArtifactMeta)
    chunk_size: int = 0
    
@dataclass
class LogChunk:
    magic: bytes = MAGIC
    lsm_key_hash: bytes = b""
    size: int = 0
    payload: bytes = b""
    
@dataclass
class TensorLogFile:
    file_id: int = 0
    path: Path = None
    size: int = 0
    

class LSMEngine(DiskBackendEngine):
    def __init__(self, *, store_dir: str, memtable_capacity: int = 1000):
        self._name = "lsm"
        self._store_dir = Path(store_dir)
        self._memtable_capacity = memtable_capacity
        
        # Initialize store directory
        if not self._store_dir.exists():
            self._store_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize memtable
        self._memtable = MemTable()
        
        self._path = self._store_dir / "lsm_log_0.dat"
        if self._path.exists():
            self._path.unlink()
        self._path.touch()

        self._file = TensorLogFile(
            file_id=0,
            path=self._path,
            size=0
        )
        

    @property
    def name(self) -> str:
        """Return the name of this engine."""
        return self._name
    
    @staticmethod
    def ensure_lsm_key(artifact_id: ArtifactId) -> None:
        """Check if the artifact_id has a valid LSM key."""
        if artifact_id.lsm_key is None:
            raise ValueError(f"Artifact {artifact_id} does not have an LSM key.")
    
    _default = object()
    def exists(self, artifact_id: ArtifactId) -> bool:
        """Check if an artifact exists in storage."""
        lsm_key = artifact_id.lsm_key
        self.ensure_lsm_key(artifact_id)
        
        exist = self._memtable.get(lsm_key, self._default) is not self._default
        if exist:
            logger.info(f"LSM key {lsm_key.hex()} exists in LSM.")
        else:
            logger.info(f"LSM key {lsm_key.hex()} does not exist in LSM.")
        return exist
    
    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        """Retrieve an artifact and its metadata from storage."""
        logger.info(f"Getting artifact {artifact_id} from LSM.")
        lsm_key = artifact_id.lsm_key
        self.ensure_lsm_key(artifact_id)
        logger.info(f"Looking up LSM key {lsm_key.hex()} in memtable.")

        value = self._memtable.get(lsm_key, self._default)
        if value is self._default:
            raise ArtifactNotFound(f"Artifact {artifact_id} not found.")

        offset = value.offset
        size = value.chunk_size
        meta = value.meta
        
        logger.info(f"Found chunk for {lsm_key.hex()} at offset {offset} with size {size} in LSM.")
        
        try:
            with self._file.path.open("r+b") as f:
                with mmap.mmap(f.fileno(), length=size, offset=offset) as m:
                    magic = m[0:4]
                    logger.info(f"Read magic header {magic} for chunk at offset {offset}.")
                    if magic != MAGIC:
                        logger.error(f"Invalid magic header at offset {offset}: expected {MAGIC}, got {magic}")
                        raise ValueError(f"Invalid magic header at offset {offset}: expected {MAGIC}, got {magic}")
                    lsm_key_hash = m[4:36]
                    if lsm_key_hash != sha256(lsm_key).digest():
                        logger.error(f"LSM key hash mismatch at offset {offset}: expected {sha256(lsm_key).hexdigest()}, got {lsm_key_hash.hex()}")
                        raise ValueError(f"LSM key hash mismatch at offset {offset}: expected {sha256(lsm_key).hexdigest()}, got {lsm_key_hash.hex()}")
                    payload_size = unpack("I", m[36:40])[0]
                    logger.info(f"Read size {payload_size} for chunk at offset {offset}.")
                    payload = bytes(m[40:40+payload_size])
                    logger.info(f"Read chunk at offset {offset} with size {payload_size} for artifact {artifact_id}.")
        except Exception:
            logger.error(f"Error reading chunk for {lsm_key.hex()} at offset {offset}")
            raise

        return payload, meta
    
    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        """Store an artifact and its metadata."""

        logger.info(f"Putting artifact {artifact_id} into LSM.")
        lsm_key = artifact_id.lsm_key
        self.ensure_lsm_key(artifact_id)
        logger.info(f"Putting {lsm_key.hex()} into LSM.")
        
        chunk = LogChunk(
            lsm_key_hash=sha256(lsm_key).digest(),
            size=len(payload),
            payload=payload
        )
        # Align offset to page boundary for mmap zero-copy reads
        page_size = mmap.ALLOCATIONGRANULARITY
        current_size = self._file.size
        aligned_offset = ((current_size + page_size - 1) // page_size) * page_size
        padding = aligned_offset - current_size

        chunk_size = 4 + len(chunk.lsm_key_hash) + 4 + len(chunk.payload)
        with self._file.path.open("ab") as f:
            if padding > 0:
                f.write(b'\x00' * padding)
            f.write(chunk.magic)
            f.write(chunk.lsm_key_hash)
            f.write(pack("I", chunk.size))
            f.write(chunk.payload)
            self._file.size = aligned_offset + chunk_size
        
        self._memtable[lsm_key] = LSMBlockValue(
            file_id=0,  # For simplicity, we use a single file in this example
            offset=aligned_offset,
            meta=meta,
            chunk_size=chunk_size,
        )
        
        logger.info(f"Wrote chunk for {lsm_key.hex()} at offset {aligned_offset} with size {chunk_size}.")
    
    def delete(self, artifact_id: ArtifactId) -> None:
        """Delete an artifact and its metadata from storage."""

        lsm_key = artifact_id.lsm_key
        self.ensure_lsm_key(artifact_id)

        if lsm_key in self._memtable:
            del self._memtable[lsm_key]
            logger.info(f"Deleted chunk for {lsm_key.hex()} from LSM.")
        else:
            raise ArtifactNotFound(f"Artifact {artifact_id} not found.")
    
    def stats(self) -> BackendStats:
        """Return statistics about the backend storage."""
        return BackendStats(
            items=len(self._memtable),
            bytes_used=sum(value.chunk_size for value in self._memtable.values()),
        )
    