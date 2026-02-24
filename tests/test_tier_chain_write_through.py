from __future__ import annotations

from stratacache.backend.cpu_store import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


def run() -> None:
    l0 = CpuMemoryLayer(store_name="l0")
    l1 = CpuMemoryLayer(store_name="l1")
    chain = TierChain(tiers=[l0, l1], links=[LinkPolicy.WRITE_THROUGH], enable_writeback_worker=False)
    try:
        aid = ArtifactId("t:wt")
        meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM, attrs={"x": 1})
        chain.store(aid, b"abc", meta)

        p0, _ = l0.get(aid)
        p1, _ = l1.get(aid)
        assert p0 == b"abc"
        assert p1 == b"abc"
    finally:
        chain.close()

