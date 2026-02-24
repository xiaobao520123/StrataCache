from __future__ import annotations

import argparse
import os
import sys
import time

from stratacache.backend.cpu_store import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="StrataCache smoke test")
    ap.add_argument("--cxl", action="store_true", help="Include CXL as tier1 (requires libcxl_shm.so + DAX)")
    ap.add_argument("--dax", default=None, help="DAX device path (exported to LMCACHE_CXL_DAX_DEVICE)")
    ap.add_argument("--writeback", action="store_true", help="Use write-back for L0->L1 (default: write-through)")
    args = ap.parse_args(argv)

    tiers = [CpuMemoryLayer(capacity_bytes=8 * 1024 * 1024, store_name="cpu0")]
    links: list[LinkPolicy] = []

    if args.cxl:
        if args.dax is not None:
            os.environ["LMCACHE_CXL_DAX_DEVICE"] = args.dax
        from stratacache.backend.cxl.store import CxlConfig, CxlMemoryLayer

        tiers.append(CxlMemoryLayer(config=CxlConfig(reset_metadata_on_init=False), store_name="cxl1"))
        links.append(LinkPolicy.WRITE_BACK if args.writeback else LinkPolicy.WRITE_THROUGH)

    chain = TierChain(tiers=tiers, links=links, enable_writeback_worker=True)
    try:
        aid = ArtifactId("demo:hello")
        meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM, attrs={"t": time.time()})
        chain.store(aid, b"hello world", meta)

        got = chain.fetch(aid, promote=True)
        assert got.payload == b"hello world"
        print(f"hit_tier={got.hit_tier} meta={got.meta.to_json()}")

        if args.writeback and args.cxl:
            # Ensure flush makes progress.
            flushed = chain.flush(aid)
            print(f"flush_count={flushed}")

        return 0
    finally:
        chain.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

