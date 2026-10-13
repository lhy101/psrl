"""A planner driven by an explicit per-boundary schedule of slot ids.

`StaticCohortPlanner` wakes every slot of one TP degree, so it cannot express an
awake set that mixes degrees. This planner names slots directly, which is what
makes a heterogeneous awake set (say one TP-2 replica beside two TP-1 replicas)
reachable, and is how the mixed-degree path is tested end to end.

The schedule is a list of slot-id lists, indexed by model version modulo its
length. Each entry must be GPU-disjoint, which `AwakeSet.validate` enforces at
switch time rather than here, so a bad entry surfaces as a rejected plan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from omegaconf import DictConfig

from psrl.workers.gen.dynamic_parallel.planner.base import (
    COHORT_NEW,
    BoundaryState,
    CohortPlanner,
    ParallelismPlan,
)
from psrl.workers.gen.dynamic_parallel.planner.registry import register_cohort_planner
from psrl.workers.gen.dynamic_parallel.slots import ReplicaSlotTable

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register_cohort_planner("scripted")
class ScriptedCohortPlanner(CohortPlanner):
    """Wakes an explicitly named set of slots, cycling by model version.

    Unlike `StaticCohortPlanner` the entries need not be homogeneous in TP, so
    this is the planner to use when the point is to serve two TP degrees at once.
    """

    def __init__(self, slot_schedule: Sequence[Sequence[int]]) -> None:
        """Bind the planner to a per-boundary schedule of slot ids.

        Args:
            slot_schedule (Sequence[Sequence[int]]): Slot-id lists to cycle through.

        Raises:
            ValueError: If the schedule is empty or holds an empty entry.
        """
        if not slot_schedule:
            raise ValueError("slot_schedule must hold at least one entry, got an empty sequence.")
        normalized = []
        for entry in slot_schedule:
            ids = tuple(int(slot_id) for slot_id in entry)
            if not ids:
                raise ValueError(f"Every slot_schedule entry must name at least one slot, got {entry!r}.")
            if len(set(ids)) != len(ids):
                raise ValueError(f"slot_schedule entry {entry!r} names the same slot twice.")
            normalized.append(ids)
        self._slot_schedule = tuple(normalized)

    @classmethod
    def from_config(cls, config: DictConfig) -> ScriptedCohortPlanner:
        """Read `slot_schedule` from the `psrl.dynamic_parallel` node.

        Args:
            config (DictConfig): The `psrl.dynamic_parallel` node.

        Returns:
            ScriptedCohortPlanner: Planner bound to the configured schedule.

        Raises:
            ValueError: If `slot_schedule` is absent.
        """
        schedule = config.get("slot_schedule", None)
        if not schedule:
            raise ValueError(
                "planner='scripted' needs psrl.dynamic_parallel.slot_schedule, a list of slot-id "
                "lists such as [[0,4,5],[2,3,4,5]]."
            )
        return cls(slot_schedule=[list(entry) for entry in schedule])

    def plan(self, table: ReplicaSlotTable, state: BoundaryState) -> ParallelismPlan:
        """Wake the slots named for this model version.

        Args:
            table (ReplicaSlotTable): Every slot the pool can run.
            state (BoundaryState): Snapshot of the boundary.

        Returns:
            ParallelismPlan: The named slots, all in one cohort.

        Raises:
            KeyError: If the schedule names a slot outside the table.
        """
        slot_ids = self._slot_schedule[state.model_version % len(self._slot_schedule)]
        slots = []
        for slot_id in slot_ids:
            slot = table.by_id.get(slot_id)
            if slot is None:
                raise KeyError(
                    f"slot_schedule names slot {slot_id}, which is not in the replica slot table "
                    f"(it holds ids 0 to {len(table) - 1})."
                )
            slots.append(slot)
        return ParallelismPlan(
            awake=frozenset(slots),
            cohort_of={slot.slot_id: COHORT_NEW for slot in slots},
        )

    def __repr__(self) -> str:
        return f"ScriptedCohortPlanner(slot_schedule={[list(e) for e in self._slot_schedule]})"
