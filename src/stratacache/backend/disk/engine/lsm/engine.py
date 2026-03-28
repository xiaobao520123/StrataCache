from stratacache.backend.disk.engine.base import DiskBackendEngine
from stratacache.backend.base import BackendStats
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound

from pathlib import Path
from typing import Tuple
import json
import math
import os
import threading
import logging

from rocksdict import Rdict, Options, SliceTransform, WriteBatch

logger = logging.getLogger(__name__)

# RocksDB key-space design:
#   - Each LSMBlockKey is a list of sub-keys (bytes).
#   - For put/get, payload is split into equal chunks, one per sub-key.
#   - Value layout per sub-key: DATA_PREFIX + sub_key -> payload_chunk
#   - Meta is stored only on the first sub-key.

META_PREFIX = b"meta:"
DATA_PREFIX = b"data:"

# Per-process singleton cache for Rdict instances (keyed by db_path)
_db_cache: dict[str, Rdict] = {}
_db_cache_lock = threading.Lock()


def _get_or_create_db(db_path: str, opts: Options) -> Rdict:
    with _db_cache_lock:
        if db_path not in _db_cache:
            _db_cache[db_path] = Rdict(db_path, opts)
        return _db_cache[db_path]


class LSMEngine(DiskBackendEngine):
    def __init__(self, *, store_dir: str, memtable_capacity: int = 1000):
        self._name = "lsm"
        self._store_dir = Path(store_dir)

        if not self._store_dir.exists():
            self._store_dir.mkdir(parents=True, exist_ok=True)

        opts = Options()
        opts.create_if_missing(True)
        opts.set_prefix_extractor(SliceTransform.create_fixed_prefix(8))  # prefix_hash (8B)
        opts.set_enable_blob_files(True)
        opts.set_min_blob_size(0)  # All values go to blob files
        opts.set_enable_blob_gc(True)
        opts.set_max_bytes_for_level_base(0x40000000)  # 1GB to avoid compactions during testing
        opts.set_target_file_size_base(0x10000000) # 256MB to keep number of files manageable
        
        

        # Use per-process subdirectory to avoid RocksDB lock conflicts
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", str(os.getpid())))
        db_path = str(self._store_dir / f"rocksdb_rank{rank}")

        self._db = _get_or_create_db(db_path, opts)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def ensure_lsm_key(artifact_id: ArtifactId) -> None:
        if artifact_id.lsm_key is None:
            raise ValueError(f"Artifact {artifact_id} does not have an LSM key.")

    def _sub_keys(self, artifact_id: ArtifactId) -> list[bytes]:
        """Return the list of sub-keys from an ArtifactId's LSMBlockKey."""
        return list(artifact_id.lsm_key)  # type: ignore[arg-type]

    def exists(self, artifact_id: ArtifactId) -> bool:
        self.ensure_lsm_key(artifact_id)
        sub_keys = self._sub_keys(artifact_id)
        data_keys = [DATA_PREFIX + sk for sk in sub_keys]
        results = self._db[data_keys]
        all_exist = all(v is not None for v in results)
        # logger.info(
        #     f"LSM key ({len(sub_keys)} sub-keys) "
        #     f"{'all exist' if all_exist else 'partially/fully missing'} in RocksDB."
        # )
        return all_exist

    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        self.ensure_lsm_key(artifact_id)
        sub_keys = self._sub_keys(artifact_id)
        data_keys = [DATA_PREFIX + sk for sk in sub_keys]

        # Batch get all sub-key payloads in one call
        chunks = self._db[data_keys]

        if any(c is None for c in chunks):
            raise ArtifactNotFound(f"Artifact {artifact_id} not found (missing sub-keys).")

        payload = b"".join(chunks)

        # Meta is stored on the first sub-key
        meta_raw = self._db.get(META_PREFIX + sub_keys[0])
        meta = ArtifactMeta.from_json(json.loads(meta_raw)) if meta_raw is not None else ArtifactMeta()

        # logger.info(f"Read artifact {artifact_id} ({len(payload)} bytes, {len(sub_keys)} sub-keys) from RocksDB.")
        return payload, meta

    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        self.ensure_lsm_key(artifact_id)
        sub_keys = self._sub_keys(artifact_id)
        n = len(sub_keys)

        # Split payload into n roughly-equal chunks
        chunk_size = math.ceil(len(payload) / n) if n > 0 else len(payload)

        # Atomic batch write for all sub-keys
        wb = WriteBatch()
        for i, sk in enumerate(sub_keys):
            wb.put(DATA_PREFIX + sk, payload[i * chunk_size : (i + 1) * chunk_size])
        wb.put(META_PREFIX + sub_keys[0], json.dumps(meta.to_json()).encode())
        self._db.write(wb)

        # logger.info(f"Wrote artifact {artifact_id} ({len(payload)} bytes, {n} sub-keys) to RocksDB.")

    def delete(self, artifact_id: ArtifactId) -> None:
        self.ensure_lsm_key(artifact_id)
        sub_keys = self._sub_keys(artifact_id)

        if self._db.get(DATA_PREFIX + sub_keys[0]) is None:
            raise ArtifactNotFound(f"Artifact {artifact_id} not found.")

        wb = WriteBatch()
        for sk in sub_keys:
            wb.delete(DATA_PREFIX + sk)
        wb.delete(META_PREFIX + sub_keys[0])
        self._db.write(wb)

        # logger.info(f"Deleted artifact {artifact_id} ({len(sub_keys)} sub-keys) from RocksDB.")

    def stats(self) -> BackendStats:
        items = 0
        bytes_used = 0
        for key, value in self._db.items():
            if key.startswith(DATA_PREFIX):
                items += 1
                bytes_used += len(value)
        return BackendStats(items=items, bytes_used=bytes_used)
    