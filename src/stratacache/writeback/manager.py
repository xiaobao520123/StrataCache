from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from stratacache.core.artifact import ArtifactId
from stratacache.tiering.policy import LinkPolicy


@dataclass(frozen=True, slots=True)
class DirtyKey:
    upper_tier: int
    artifact: str


class WritebackManager:
    """
    Minimal write-back manager for tier chains.

    It tracks dirty artifacts per upper tier index where the link to the next
    tier is WRITE_BACK, and flushes them via a hop-level callback:

      flush_hop(upper_tier_index, artifact_id)
    """

    def __init__(
        self,
        *,
        links: list[LinkPolicy],
        flush_hop: Callable[[int, ArtifactId], None],
        worker_name: str = "writeback",
        enable_worker: bool = True,
        idle_sleep_s: float = 0.01,
        retry_delay_s: float = 0.1,
    ) -> None:
        self._links = links
        self._flush_hop = flush_hop
        self._idle_sleep_s = float(idle_sleep_s)
        self._retry_delay_s = float(retry_delay_s)

        self._dirty: set[DirtyKey] = set()
        self._q: "queue.Queue[DirtyKey]" = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._attempts: dict[DirtyKey, int] = {}

        self._thread: Optional[threading.Thread] = None
        if enable_worker:
            self._thread = threading.Thread(target=self._run, name=worker_name, daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def mark_dirty(self, upper_tier: int, artifact_id: ArtifactId) -> None:
        if upper_tier < 0 or upper_tier >= len(self._links):
            return
        if self._links[upper_tier] != LinkPolicy.WRITE_BACK:
            return
        dk = DirtyKey(upper_tier=upper_tier, artifact=str(artifact_id))
        with self._lock:
            if dk in self._dirty:
                return
            self._dirty.add(dk)
            self._q.put(dk)

    def clear_dirty(self, upper_tier: int, artifact_id: ArtifactId) -> None:
        dk = DirtyKey(upper_tier=upper_tier, artifact=str(artifact_id))
        with self._lock:
            self._dirty.discard(dk)
            self._attempts.pop(dk, None)

    def is_dirty(self, upper_tier: int, artifact_id: ArtifactId) -> bool:
        dk = DirtyKey(upper_tier=upper_tier, artifact=str(artifact_id))
        with self._lock:
            return dk in self._dirty

    def flush(self, artifact_id: Optional[ArtifactId] = None, *, max_items: Optional[int] = None) -> int:
        """
        Synchronous best-effort flush.

        - If artifact_id is set, flushes that artifact across all write-back links.
        - Else, drains up to max_items from the dirty set.
        """
        flushed = 0
        if artifact_id is not None:
            for upper in range(len(self._links)):
                if self._links[upper] == LinkPolicy.WRITE_BACK and self.is_dirty(upper, artifact_id):
                    self._flush_one(DirtyKey(upper_tier=upper, artifact=str(artifact_id)))
                    flushed += 1
            return flushed

        # Drain snapshot to avoid holding lock while flushing.
        with self._lock:
            keys = list(self._dirty)
        if max_items is not None:
            keys = keys[: int(max_items)]
        for dk in keys:
            self._flush_one(dk)
            flushed += 1
        return flushed

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                dk = self._q.get(timeout=0.05)
            except queue.Empty:
                time.sleep(self._idle_sleep_s)
                continue
            self._flush_one(dk)

    def _flush_one(self, dk: DirtyKey) -> None:
        # If already cleared, skip.
        with self._lock:
            if dk not in self._dirty:
                return
        artifact_id = ArtifactId(dk.artifact)
        try:
            self._flush_hop(dk.upper_tier, artifact_id)
        except Exception:
            # Best-effort retry: keep it dirty and re-enqueue with a small delay.
            with self._lock:
                self._attempts[dk] = self._attempts.get(dk, 0) + 1
                still_dirty = dk in self._dirty
            if still_dirty and not self._stop.is_set():
                time.sleep(self._retry_delay_s)
                self._q.put(dk)
            raise

