from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stratacache.core.artifact import ArtifactId


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """
    v0.1 placeholder.

    Future: describe which artifacts to promote/demote between tiers.
    """

    promote: tuple[ArtifactId, ...] = ()
    demote: tuple[ArtifactId, ...] = ()


class MigrationPlanner:
    """
    v0.1 stub.

    Later this will inspect tier stats/pressure and produce a MigrationPlan.
    """

    def plan(self, *, candidates: Iterable[ArtifactId]) -> MigrationPlan:
        _ = list(candidates)
        return MigrationPlan()

