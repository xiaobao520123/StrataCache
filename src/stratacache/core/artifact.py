from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import math
import time


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
    
    def __init__(self, value: str):
        object.__setattr__(self, "value", value)

    def __str__(self) -> str:
        return self.value


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

@dataclass()
class ArtifactHeat:
    artifact_id: ArtifactId = field(default_factory=lambda: ArtifactId("default_artifact"))
    frequency: int = 0
    length: int = 0
    ts_last_accessed: float = 0.0
    ts_peak: float = 0.0


def compute_artifact_heat(artifact: ArtifactHeat, periodic: bool) -> float:
    ts = time.time()
    base = math.log(1 + artifact.frequency / artifact.length)
    decay_rate = 0.05
    recency = math.exp(-decay_rate * (ts - artifact.ts_last_accessed))

    if not periodic:
        return base * recency

    alpha = 3.0
    periodic = 1 + alpha * 0.5 * (1 + math.cos(math.pi / 12 * (ts - artifact.ts_peak)))
    return base * recency * periodic


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: ArtifactId
    payload: bytes
    meta: ArtifactMeta = field(default_factory=ArtifactMeta)


if __name__ == "__main__":
    artifact_heat = ArtifactHeat(
        artifact_id=ArtifactId("example_artifact"),
        frequency=20,
        length=1024,
        ts_last_accessed=time.time() - 60,
        ts_peak=time.time() - 3600,
    )
    heat = compute_artifact_heat(artifact_heat, periodic=True)
    print(f"Artifact heat: {heat}")

    artifact_heat = ArtifactHeat(
        artifact_id=ArtifactId("example_artifact"),
        frequency=500,
        length=100,
        ts_last_accessed=time.time() - 60,
        ts_peak=time.time() - 3600,
    )
    heat = compute_artifact_heat(artifact_heat, periodic=True)
    print(f"Artifact heat: {heat}")