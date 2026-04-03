from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ArtifactType(str, Enum):
    KV_BLOCKS = "kv_blocks"
    PARAM_CHUNK = "param_chunk"
    OPTIMIZER_CHUNK = "optimizer_chunk"
    TENSOR_SHARD = "tensor_shard"
    MOE_WEIGHTS = "moe_weights"
    IMAGE_KV = "image_kv"
    MEMORY_ENTRY = "memory_entry"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ArtifactId:
    """
    Stable identity for a stored artifact.

    v0.1 uses a string id (caller is responsible for namespacing).
    """

    value: str

    def __str__(self) -> str:
        return self.value
    
    @property
    def chunk_end(self) -> int:
        return int(self.value.split(":")[2].replace("chunk_end=", ""))


JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """
    JSON-serializable metadata.

    Keep this engine-agnostic; put engine-specific bits under `engine_hints`.
    """

    artifact_type: ArtifactType = ArtifactType.CUSTOM
    engine_hints: JsonDict = field(default_factory=dict)
    attrs: JsonDict = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        return {
            "artifact_type": self.artifact_type.value,
            "engine_hints": self.engine_hints,
            "attrs": self.attrs,
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "ArtifactMeta":
        at = d.get("artifact_type", ArtifactType.CUSTOM)
        if isinstance(at, str):
            artifact_type = ArtifactType(at)
        else:
            artifact_type = ArtifactType.CUSTOM
        return ArtifactMeta(
            artifact_type=artifact_type,
            engine_hints=dict(d.get("engine_hints", {})),
            attrs=dict(d.get("attrs", {})),
        )


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: ArtifactId
    payload: bytes
    meta: ArtifactMeta = field(default_factory=ArtifactMeta)
