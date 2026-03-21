from __future__ import annotations

from collections.abc import Sequence

from stratacache.backend.base import MemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound
from stratacache.engine.types import AccessMode, ContainsResult, LoadResult
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy

import time

class StorageEngine:
    """
    Public storage facade used by adapters and direct clients.

    It exposes a small, stable API (`store/load/contains/delete`) and hides
    tier traversal details behind `AccessMode`.
    """

    def __init__(self, chain: TierChain) -> None:
        self._chain = chain

    @classmethod
    def from_tiers(
        cls,
        *,
        tiers: Sequence[MemoryLayer],
        links: Sequence[LinkPolicy],
        enable_writeback_worker: bool = True,
    ) -> "StorageEngine":
        """Build an engine directly from memory layers and link policies."""
        chain = TierChain(
            tiers=tiers,
            links=links,
            enable_writeback_worker=enable_writeback_worker,
        )
        return cls(chain)

    @property
    def tier_names(self) -> list[str]:
        return self._chain.tier_names

    @property
    def chain(self) -> TierChain:
        """Expose the underlying TierChain for advanced/internal integrations."""
        return self._chain

    def close(self) -> None:
        """Stop background workers and release chain resources."""
        self._chain.close()

    def store(
        self,
        artifact_id: ArtifactId,
        payload: bytes,
        meta: ArtifactMeta,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
    ) -> None:
        """
        Store an artifact.

        - `medium=None`: write through the chain head.
        - `mode=exact` with `medium`: write only to that tier.
        - `mode=chain` with `medium`: write at that tier and propagate downstream.
        """
        m = _normalize_mode(mode)
        if medium is None:
            self._chain.store(artifact_id, payload, meta)
            return

        if m == AccessMode.CHAIN:
            self._chain.store_at(medium, artifact_id, payload, meta, propagate=True)
            return
        if m == AccessMode.EXACT:
            self._chain.store_at(medium, artifact_id, payload, meta, propagate=False)
            return
        if m == AccessMode.PREFER:
            try:
                self._chain.store_at(medium, artifact_id, payload, meta, propagate=False)
            except ValueError:
                self._chain.store(artifact_id, payload, meta)
            return
        raise ValueError(f"unknown mode: {mode}")

    def load(
        self,
        artifact_id: ArtifactId,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
        promote: bool = True,
    ) -> LoadResult:
        """
        Load an artifact with tier selection behavior controlled by `mode`.

        - `chain`: normal top-down lookup.
        - `exact`: only read from `medium`.
        - `prefer`: try `medium`, fallback to chain lookup.
        """
        m = _normalize_mode(mode)

        if medium is None or m == AccessMode.CHAIN:
            perf_counter_start = time.perf_counter()
            fr = self._chain.fetch(artifact_id, promote=promote)
            latency_ms = (time.perf_counter() - perf_counter_start) * 1000
            self._telemetry.on_tier_ops_async(
                device_type=StrataTierType.CPU,
                op_type="load",
                latency_ms=latency_ms,
                size=len(fr.payload) if fr.payload is not None else 0,
            )
            return LoadResult(
                payload=fr.payload,
                meta=fr.meta,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        if m == AccessMode.EXACT:
            perf_counter_start = time.perf_counter()
            fr = self._chain.fetch_from(medium, artifact_id, promote=promote)
            latency_ms = (time.perf_counter() - perf_counter_start) * 1000
            self._telemetry.on_tier_ops_async(
                device_type=StrataTierType.CPU,
                op_type="load",
                latency_ms=latency_ms,
                size=len(fr.payload) if fr.payload is not None else 0,
            )
            return LoadResult(
                payload=fr.payload,
                meta=fr.meta,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        if m == AccessMode.PREFER:
            try:
                perf_counter_start = time.perf_counter()
                fr = self._chain.fetch_from(medium, artifact_id, promote=promote)
                latency_ms = (time.perf_counter() - perf_counter_start) * 1000
            except (ArtifactNotFound, ValueError):
                perf_counter_start = time.perf_counter()
                fr = self._chain.fetch(artifact_id, promote=promote)
                latency_ms = (time.perf_counter() - perf_counter_start) * 1000
            self._telemetry.on_tier_ops_async(
                device_type=StrataTierType.CPU,
                op_type="load",
                latency_ms=latency_ms,
                size=len(fr.payload) if fr.payload is not None else 0,
            )
            return LoadResult(
                payload=fr.payload,
                meta=fr.meta,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        raise ValueError(f"unknown mode: {mode}")

    def contains(
        self,
        artifact_id: ArtifactId,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
    ) -> ContainsResult:
        """Check existence and return where the artifact was found."""
        m = _normalize_mode(mode)

        if medium is None or m == AccessMode.CHAIN:
            hit = self._chain.exists(artifact_id)
            if hit is None:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            return ContainsResult(
                exists=True,
                hit_tier=hit,
                hit_medium=self._chain.tier_names[hit],
            )

        if m == AccessMode.EXACT:
            try:
                ok = self._chain.exists_in(medium, artifact_id)
            except ValueError:
                ok = False
            if not ok:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            idx = _resolve_tier(self._chain, medium)
            return ContainsResult(exists=True, hit_tier=idx, hit_medium=self._chain.tier_names[idx])

        if m == AccessMode.PREFER:
            try:
                ok = self._chain.exists_in(medium, artifact_id)
            except ValueError:
                ok = False
            if ok:
                idx = _resolve_tier(self._chain, medium)
                return ContainsResult(
                    exists=True,
                    hit_tier=idx,
                    hit_medium=self._chain.tier_names[idx],
                )
            hit = self._chain.exists(artifact_id)
            if hit is None:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            return ContainsResult(exists=True, hit_tier=hit, hit_medium=self._chain.tier_names[hit])

        raise ValueError(f"unknown mode: {mode}")

    def delete(self, artifact_id: ArtifactId, *, medium: int | str | None = None) -> None:
        """Delete from one tier (`medium`) or all tiers (`medium=None`)."""
        if medium is None:
            self._chain.delete(artifact_id)
            return
        self._chain.delete_from(medium, artifact_id)


def _normalize_mode(mode: AccessMode | str) -> AccessMode:
    if isinstance(mode, AccessMode):
        return mode
    s = str(mode).strip().lower()
    try:
        return AccessMode(s)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"unknown mode: {mode}") from e


def _resolve_tier(chain: TierChain, tier: int | str) -> int:
    if isinstance(tier, int):
        if tier < 0 or tier >= len(chain.tiers):
            raise ValueError(f"unknown tier: {tier}")
        return tier
    name = str(tier)
    for i, t in enumerate(chain.tiers):
        if t.name == name:
            return i
    raise ValueError(f"unknown tier: {tier}")
