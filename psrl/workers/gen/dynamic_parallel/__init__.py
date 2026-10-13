"""Dynamic heterogeneous rollout parallelism.

Every configured TP degree gets its replica slots materialized at startup, and a
weight-update boundary switches which GPU-disjoint subset of them is awake. The
NIXL shard-grid negotiation is a one-shot barrier, so a TP degree absent at
startup can never be admitted later. Pre-materializing is what makes switching
possible, and it is why no `RolloutInstanceId` ever has to be renumbered.

`slots` and `awake_set` are pure. `switch` owns the Ray and SMG calls.
"""

from psrl.workers.gen.dynamic_parallel.awake_set import AwakeSet, SlotTransition
from psrl.workers.gen.dynamic_parallel.planner import (
    COHORT_NEW,
    COHORT_OLD,
    BoundaryState,
    CohortPlanner,
    ParallelismPlan,
    build_cohort_planner,
    register_cohort_planner,
)
from psrl.workers.gen.dynamic_parallel.slots import (
    NodeGPULayout,
    ReplicaSlot,
    ReplicaSlotTable,
    build_slot_table,
)

__all__ = [
    "COHORT_NEW",
    "COHORT_OLD",
    "AwakeSet",
    "BoundaryState",
    "CohortPlanner",
    "NodeGPULayout",
    "ParallelismPlan",
    "ReplicaSlot",
    "ReplicaSlotTable",
    "SlotTransition",
    "build_cohort_planner",
    "build_slot_table",
    "register_cohort_planner",
]
