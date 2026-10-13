"""SwitchMixin: apply a target awake set at a weight-update boundary.

This is the only module in `dynamic_parallel` that touches Ray and SMG. It turns a
`SlotTransition` from `AwakeSet` into real sleep, wake, and weight-pull calls, in
the one order that works: outgoing slots must release their GPU pages before the
incoming slots on those GPUs can map them.

The mixin composes into `RolloutCoordinator` alongside `CommandHandlerMixin` and
the sync strategy. It is a coordinator method rather than a new `CommandType`,
following the direction the SYNC command was retired in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from psrl.utils.server.command import Command, CommandType
from psrl.workers.gen.dynamic_parallel.awake_set import AwakeSet, SlotTransition
from psrl.workers.gen.dynamic_parallel.slots import ReplicaSlot
from psrl.workers.gen.smg_adapter import build_pause_resume_payload
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class SwitchMixin:
    """Provides ``switch_to`` for RolloutCoordinator.

    Expects ``self`` to carry ``awake_set``, ``slot_to_replica_id``,
    ``server_handles``, ``instance_ids``, and the ``_gateway_post_json``,
    ``_set_routing_loop_running``, ``_do_sleep_instance``,
    ``_do_wake_up_instance``, ``_quarantine_failed_replicas``, ``sync_with_ps``,
    and ``exec_command`` members provided by ``CoordinatorBase`` and
    ``SyncAndMigrateMixin``.
    """

    def _slot_replica_ids(self, slots: tuple[ReplicaSlot, ...]) -> list[str]:
        """Map slots to their SMG replica ids, skipping any not yet registered."""
        replica_ids = []
        for slot in slots:
            replica_id = self.slot_to_replica_id.get(slot.slot_id)
            if replica_id is None:
                psrl_logger.warning(f"Slot {slot.slot_id} has no registered replica id. Skipping it.")
                continue
            replica_ids.append(replica_id)
        return replica_ids

    def _slot_instance_ids(self, slots: tuple[ReplicaSlot, ...]) -> list[RolloutInstanceId]:
        """Expand slots to every `RolloutInstanceId` their replicas own."""
        replica_ids = set(self._slot_replica_ids(slots))
        return sorted(instance_id for instance_id in self.instance_ids if instance_id[0] in replica_ids)

    async def switch_to(self, target: frozenset[ReplicaSlot], abort_in_flight: bool = True) -> SlotTransition | None:
        """Move the awake set to `target`, sleeping outgoing slots before waking incoming ones.

        Args:
            target: Slots that should be awake afterwards. Must be GPU-disjoint.
            abort_in_flight: Abort requests on outgoing slots. Partial rollout
                preserves their prefixes, and under premise A2 the KV is discarded
                at the boundary anyway, so aborting costs only the re-prefill that
                was already due.

        Returns:
            The applied transition, or None when nothing had to move.

        Raises:
            KeyError: If `target` names a slot outside the table.
            ValueError: If `target` is not GPU-disjoint.
        """
        awake_set: AwakeSet = self.awake_set
        transition = awake_set.plan_transition(target)
        if transition.is_empty:
            psrl_logger.warning(
                f"Awake set already matches the target {sorted(awake_set.current_ids())}. No switch needed."
            )
            return None

        started = time.perf_counter()
        sleep_replica_ids = self._slot_replica_ids(transition.to_sleep)
        wake_replica_ids = self._slot_replica_ids(transition.to_wake)
        psrl_logger.warning(
            f"Switching awake slots: sleep={[s.slot_id for s in transition.to_sleep]}, "
            f"wake={[s.slot_id for s in transition.to_wake]}, "
            f"keep={[s.slot_id for s in transition.unchanged]}. "
            f"Resolved replica ids: sleep={sleep_replica_ids!r}, wake={wake_replica_ids!r}."
        )

        # Routing must stop before any slot loses its weights, or SMG would dispatch
        # to an engine that is on its way down.
        await self._set_routing_loop_running(False)
        try:
            if sleep_replica_ids:
                sleep_instance_ids = self._slot_instance_ids(transition.to_sleep)
                await self._gateway_post_json("/workers/pause", payload=build_pause_resume_payload(sleep_instance_ids))
                if abort_in_flight and sleep_instance_ids:
                    await self.exec_command(
                        Command(type=CommandType.ABORT, instance_ids=sleep_instance_ids),
                        blocking=True,
                    )
                await asyncio.gather(*[self._do_sleep_instance(replica_id) for replica_id in sleep_replica_ids])

            if wake_replica_ids:
                await asyncio.gather(*[self._do_wake_up_instance(replica_id) for replica_id in wake_replica_ids])
        except Exception:
            # Leave the awake set uncommitted so the next boundary replans from the
            # last state that was actually realized on the GPUs.
            await self._quarantine_failed_replicas(wake_replica_ids, self._slot_instance_ids(transition.to_wake))
            await self._set_routing_loop_running(True)
            psrl_logger.exception("Switch failed. The awake set is unchanged.")
            raise

        # A level-2 wake leaves weights unmapped, so pulling is mandatory rather than
        # an optimization. `sync_with_ps` also republishes the weight version to SMG.
        if wake_replica_ids:
            wake_instance_ids = self._slot_instance_ids(transition.to_wake)
            await self.sync_with_ps(
                wake_instance_ids,
                wait_model_sync=True,
                wait_interrupted_partial_requests_loop_back=False,
            )
            await self._gateway_post_json("/workers/resume", payload=build_pause_resume_payload(wake_instance_ids))

        await self._set_routing_loop_running(True)
        awake_set.commit(transition)
        psrl_logger.warning(
            f"Switch complete in {time.perf_counter() - started:.2f}s. "
            f"Awake slots are now {sorted(awake_set.current_ids())}."
        )
        return transition

    async def replan_parallelism(self, model_version: int) -> SlotTransition | None:
        """Re-solve the awake set for a new model version and apply it.

        Called once per weight-update boundary. Returns None when dynamic parallelism
        is off, when no planner is installed, or when the plan matches the current
        awake set.

        A failure here is logged and swallowed rather than raised, because the awake
        set stays uncommitted on failure and the next boundary replans from the last
        realized state. Letting it propagate would abort training over a parallelism
        decision that the run can continue without.
        """
        if self.awake_set is None or self.cohort_planner is None:
            psrl_logger.warning(
                f"replan_parallelism: skipped at version {model_version} "
                f"(awake_set={self.awake_set is not None}, planner={self.cohort_planner!r})."
            )
            return None
        if not self.config.psrl.dynamic_parallel.switch_on_boundary:
            psrl_logger.warning(
                f"replan_parallelism: skipped at version {model_version} (switch_on_boundary=False)."
            )
            return None

        psrl_logger.warning(
            f"replan_parallelism: planning at version {model_version}, "
            f"current awake={sorted(self.awake_set.current_ids())}."
        )
        from psrl.workers.gen.dynamic_parallel.planner import BoundaryState

        state = BoundaryState(
            model_version=model_version,
            staleness_budget=self.config.psrl.staleness,
            replica_stats={slot.slot_id: self._slot_engine_snapshot(slot) for slot in self.awake_set.current()},
        )
        try:
            plan = self.cohort_planner.plan(self.awake_set.table, state)
            psrl_logger.warning(
                f"replan_parallelism: plan={plan!r}, calling switch_to."
            )
            self.slot_to_cohort = dict(plan.cohort_of)
            return await self.switch_to(plan.awake)
        except Exception:
            psrl_logger.exception(
                f"Parallelism replan failed at version {model_version}. Keeping the current awake set."
            )
            return None

    def _slot_engine_snapshot(self, slot: ReplicaSlot) -> dict:
        """Latest engine stats for a slot, or an empty dict when it has none yet."""
        replica_id = self.slot_to_replica_id.get(slot.slot_id)
        if replica_id is None:
            return {}
        for (candidate_id, _), stats in self.instance_to_engine_status.items():
            if candidate_id == replica_id:
                return stats.snapshot if hasattr(stats, "snapshot") else {}
        return {}
