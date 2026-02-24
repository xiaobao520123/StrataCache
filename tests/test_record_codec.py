from __future__ import annotations

from stratacache.core.artifact import ArtifactMeta, ArtifactType
from stratacache.core.record_codec import decode_record, encode_record


def run() -> None:
    payload = b"\x00\x01hello"
    meta = ArtifactMeta(artifact_type=ArtifactType.MEMORY_ENTRY, attrs={"a": 1, "b": "x"})
    buf = encode_record(payload, meta)
    out_payload, out_meta = decode_record(buf)
    assert out_payload == payload
    assert out_meta.artifact_type == ArtifactType.MEMORY_ENTRY
    assert out_meta.attrs["a"] == 1
    assert out_meta.attrs["b"] == "x"

