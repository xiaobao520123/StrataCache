from __future__ import annotations

from stratacache.backend.cpu_store import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.core.errors import ArtifactNotFound
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


def run() -> None:
    # Single-hop write-back: L0 dirty until flush.
    l0 = CpuMemoryLayer(store_name="l0")
    l1 = CpuMemoryLayer(store_name="l1")
    chain = TierChain(tiers=[l0, l1], links=[LinkPolicy.WRITE_BACK], enable_writeback_worker=False)
    try:
        aid = ArtifactId("t:wb1")
        meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM)
        chain.store(aid, b"v1", meta)

        assert l0.get(aid)[0] == b"v1"
        try:
            l1.get(aid)
            assert False, "expected miss in lower tier before flush"
        except ArtifactNotFound:
            pass

        n = chain.flush(aid)
        assert n >= 1
        assert l1.get(aid)[0] == b"v1"
    finally:
        chain.close()

    # Multi-hop write-back: flush(id) should converge (after TierChain.flush fix).
    a = CpuMemoryLayer(store_name="a")
    b = CpuMemoryLayer(store_name="b")
    c = CpuMemoryLayer(store_name="c")
    chain2 = TierChain(
        tiers=[a, b, c],
        links=[LinkPolicy.WRITE_BACK, LinkPolicy.WRITE_BACK],
        enable_writeback_worker=False,
    )
    try:
        aid2 = ArtifactId("t:wb2")
        meta2 = ArtifactMeta(artifact_type=ArtifactType.CUSTOM)
        chain2.store(aid2, b"v2", meta2)
        chain2.flush(aid2)
        assert b.get(aid2)[0] == b"v2"
        assert c.get(aid2)[0] == b"v2"
    finally:
        chain2.close()

