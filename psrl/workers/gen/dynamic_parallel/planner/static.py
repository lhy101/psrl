"""A planner that holds one TP degree awake for the whole job.

This is the behavior of a run without dynamic switching, expressed as a policy
so the mechanism can be exercised and measured before any solver exists. A
config-driven schedule makes it alternate across boundaries, which is how the
switch path is tested end to end.
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


@register_cohort_planner("static")
class StaticCohortPlanner(CohortPlanner):
    """Wakes every slot of one TP degree, optionally cycling degrees per boundary.

    With a single-entry schedule the awake set never changes, which is the
    baseline a dynamic policy must beat. With several entries it walks the
    schedule by model version, which drives the switch path without needing a
    solver.
    """

    def __init__(self, tp_schedule: Sequence[int]) -> None:
        """Bind the planner to a TP schedule indexed by model version.

        Args:
            tp_schedule (Sequence[int]): TP degrees to cycle through.

        Raises:
            ValueError: If the schedule is empty or holds a non-positive degree.
        """
        if not tp_schedule:
            raise ValueError("tp_schedule must name at least one TP degree, got an empty sequence.")
        for tp in tp_schedule:
            if tp < 1:
                raise ValueError(f"TP degrees in tp_schedule must be positive, got {tp!r}.")
        self._tp_schedule = tuple(tp_schedule)

    @classmethod
    def from_config(cls, config: DictConfig) -> StaticCohortPlanner:
        """Read `tp_schedule`, falling back to the first entry of `tp_degrees`.

        Args:
            config (DictConfig): The `psrl.dynamic_parallel` node.

        Returns:
            StaticCohortPlanner: Planner bound to the configured schedule.
        """
        schedule = config.get("tp_schedule", None) or [config.get("tp_degrees", [1])[0]]
        return cls(tp_schedule=list(schedule))

    def plan(self, table: ReplicaSlotTable, state: BoundaryState) -> ParallelismPlan:
        """Wake every slot of the scheduled TP degree for this model version.

        Args:
            table (ReplicaSlotTable): Every slot the pool can run.
            state (BoundaryState): Snapshot of the boundary.

        Returns:
            ParallelismPlan: Every slot of one TP degree, all in one cohort.

        Raises:
            ValueError: If the scheduled degree has no slots in the table.
        """
        tp = self._tp_schedule[state.model_version % len(self._tp_schedule)]
        slots = table.by_tp.get(tp)
        if not slots:
            raise ValueError(
                f"TP degree {tp} from the schedule is not in the replica slot table, which holds "
                f"{list(table.tp_degrees)}."
            )
        # A static plan draws no old/new distinction, so every slot serves one cohort.
        return ParallelismPlan(
            awake=frozenset(slots),
            cohort_of={slot.slot_id: COHORT_NEW for slot in slots},
        )

    def __repr__(self) -> str:
        return f"StaticCohortPlanner(tp_schedule={list(self._tp_schedule)})"
