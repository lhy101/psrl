"""The decision seam: which replica slots should be awake after a boundary.

A planner reads the state at a weight-update boundary and names a target awake
set. It performs no I/O and holds no Ray handles, so a policy can be tested
against a synthetic `BoundaryState` without a cluster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from omegaconf import DictConfig

from psrl.workers.gen.dynamic_parallel.slots import ReplicaSlot, ReplicaSlotTable

# Cohort labels. A cohort is a group of trajectories that wants the same
# parallelism, so `OLD` trajectories carry staleness and want low latency while
# `NEW` ones want aggregate throughput.
COHORT_OLD = "old"
COHORT_NEW = "new"


@dataclass(frozen=True)
class ParallelismPlan:
    """A target awake set plus the cohort each awake slot serves.

    `cohort_of` is keyed by `slot_id` and covers exactly the awake slots. The
    switch applies `awake`, and request steering uses `cohort_of` to decide
    which trajectories may route to which slot.
    """

    awake: frozenset[ReplicaSlot]
    cohort_of: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        awake_ids = {slot.slot_id for slot in self.awake}
        labelled = set(self.cohort_of)
        if labelled - awake_ids:
            raise ValueError(f"cohort_of labels slots {sorted(labelled - awake_ids)} that are not in the awake set.")
        if awake_ids - labelled:
            raise ValueError(f"cohort_of is missing labels for awake slots {sorted(awake_ids - labelled)}.")

    def slots_for_cohort(self, cohort: str) -> frozenset[ReplicaSlot]:
        """Return the awake slots serving one cohort.

        Args:
            cohort (str): A cohort label such as `COHORT_OLD`.

        Returns:
            frozenset[ReplicaSlot]: Awake slots labelled with that cohort.
        """
        return frozenset(slot for slot in self.awake if self.cohort_of[slot.slot_id] == cohort)

    def __repr__(self) -> str:
        by_cohort = sorted((self.cohort_of[s.slot_id], s.slot_id) for s in self.awake)
        return f"ParallelismPlan({by_cohort})"


@dataclass(frozen=True)
class BoundaryState:
    """What the coordinator knows at a weight-update boundary.

    This is a snapshot, not a live view. A planner must not assume any value
    stays current while it runs.
    """

    model_version: int
    staleness_budget: int
    """Max model-version lag a trajectory may carry and still count, `S` in the SCG metric."""
    in_flight_by_cohort: dict[str, int] = field(default_factory=dict)
    replica_stats: dict[int, dict] = field(default_factory=dict)
    """Latest engine stats per awake `slot_id`. Sleeping slots are absent, not zeroed."""


class CohortPlanner(ABC):
    """Decides the target awake set at each weight-update boundary."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: DictConfig) -> CohortPlanner:
        """Build this planner from its config node.

        Args:
            config (DictConfig): The `psrl.dynamic_parallel` node.

        Returns:
            CohortPlanner: A planner ready to be called at each boundary.
        """

    @abstractmethod
    def plan(self, table: ReplicaSlotTable, state: BoundaryState) -> ParallelismPlan:
        """Name the slots that should be awake after this boundary.

        The returned set must be GPU-disjoint. `AwakeSet.validate` enforces this,
        so a planner bug surfaces as a rejected plan rather than as two replicas
        fighting over one GPU.

        Args:
            table (ReplicaSlotTable): Every slot the pool can run.
            state (BoundaryState): Snapshot of the boundary.

        Returns:
            ParallelismPlan: Target awake set with a cohort label per slot.
        """
