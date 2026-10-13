"""Which replica slots are live, and how to move between live sets.

A switch never renumbers anything. It moves the awake set from one
GPU-disjoint subset of the slot table to another, so every `RolloutInstanceId`
stays valid for the life of the job.

This module is pure state plus validation. `switch.py` owns the Ray and SMG
calls that make a planned transition real.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from psrl.workers.gen.dynamic_parallel.slots import ReplicaSlot, ReplicaSlotTable

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class SlotTransition:
    """The slots to sleep and wake to reach a target awake set.

    `to_sleep` must be fully asleep before `to_wake` starts, because a GPU's
    physical pages are only free once the outgoing slot has released them.
    `unchanged` slots are already awake and stay untouched, which is what makes
    a partial switch cheaper than a full one.
    """

    to_sleep: tuple[ReplicaSlot, ...]
    to_wake: tuple[ReplicaSlot, ...]
    unchanged: tuple[ReplicaSlot, ...]

    @property
    def is_empty(self) -> bool:
        """Whether this transition would change nothing."""
        return not self.to_sleep and not self.to_wake

    def __repr__(self) -> str:
        return (
            f"SlotTransition(sleep={[s.slot_id for s in self.to_sleep]}, "
            f"wake={[s.slot_id for s in self.to_wake]}, "
            f"keep={[s.slot_id for s in self.unchanged]})"
        )


class AwakeSet:
    """The set of replica slots currently serving requests.

    Starts empty. The caller commits a transition only after the sleep and wake
    calls it describes have succeeded, so a failed switch leaves this object
    reporting the last state that was actually realized on the GPUs.
    """

    def __init__(self, table: ReplicaSlotTable) -> None:
        self._table = table
        self._awake: frozenset[ReplicaSlot] = frozenset()

    @property
    def table(self) -> ReplicaSlotTable:
        """The slot table this awake set draws from."""
        return self._table

    def current(self) -> frozenset[ReplicaSlot]:
        """Return the slots that are awake right now."""
        return self._awake

    def current_ids(self) -> frozenset[int]:
        """Return the ids of the slots that are awake right now."""
        return frozenset(slot.slot_id for slot in self._awake)

    def validate(self, target: Iterable[ReplicaSlot]) -> frozenset[ReplicaSlot]:
        """Check that a target set is nameable and internally GPU-disjoint.

        Args:
            target (Iterable[ReplicaSlot]): The slots that should be awake after the switch.

        Returns:
            frozenset[ReplicaSlot]: The validated target.

        Raises:
            KeyError: If a slot is not in this table. `slot_id` is only unique
                within one table, so a slot built from a different layout is
                rejected on identity rather than on its id alone.
            ValueError: If two slots in the target share a GPU.
        """
        target = frozenset(target)
        for slot in target:
            known = self._table.by_id.get(slot.slot_id)
            if known is None:
                raise KeyError(
                    f"Slot id {slot.slot_id} is not in the replica slot table "
                    f"(it holds ids 0 to {len(self._table) - 1})."
                )
            if known != slot:
                raise KeyError(
                    f"Slot {slot!r} is not in the replica slot table. Its id {slot.slot_id} is held by "
                    f"{known!r}, so this slot came from a different layout and would alias the wrong GPUs."
                )

        ordered = sorted(target, key=lambda s: s.slot_id)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if self._table.conflicts(a, b):
                    raise ValueError(
                        f"Slots {a.slot_id} and {b.slot_id} share a GPU and cannot be awake together ({a!r} vs {b!r})."
                    )
        return target

    def plan_transition(self, target: Iterable[ReplicaSlot]) -> SlotTransition:
        """Compute the minimal sleep and wake sets to reach `target`.

        Args:
            target (Iterable[ReplicaSlot]): The slots that should be awake after the switch.

        Returns:
            SlotTransition: Slots to sleep, to wake, and to leave untouched.

        Raises:
            KeyError: If `target` names a slot absent from this table.
            ValueError: If `target` is not internally GPU-disjoint.
        """
        target = self.validate(target)
        return SlotTransition(
            to_sleep=tuple(sorted(self._awake - target, key=lambda s: s.slot_id)),
            to_wake=tuple(sorted(target - self._awake, key=lambda s: s.slot_id)),
            unchanged=tuple(sorted(self._awake & target, key=lambda s: s.slot_id)),
        )

    def commit(self, transition: SlotTransition) -> None:
        """Record a transition whose sleep and wake calls have already succeeded.

        Args:
            transition (SlotTransition): The transition that was realized on the GPUs.

        Raises:
            KeyError: If the transition names a slot absent from this table.
            ValueError: If the resulting set is not GPU-disjoint, which means the
                transition was not built by `plan_transition` against this state.
        """
        updated = (self._awake - set(transition.to_sleep)) | set(transition.to_wake)
        self._awake = self.validate(updated)
        psrl_logger.info(f"Awake replica slots are now {sorted(self.current_ids())} after {transition!r}.")
