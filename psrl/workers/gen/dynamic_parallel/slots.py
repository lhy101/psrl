"""Replica slots: the fixed set of vLLM replicas the rollout pool can run.

A `ReplicaSlot` is one replica's permanent reservation, a TP degree bound to a
contiguous GPU range on one node. Every slot for every configured TP degree is
materialized at startup because the NIXL shard-grid negotiation is a one-shot
barrier, so a TP degree absent at that barrier can never be admitted later.
Slots over the same GPUs therefore coexist and take turns being awake.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from psrl.utils.gpu_topology import GPUSlot

# Node id -> the GPU slots that node contributes to the rollout pool.
NodeGPULayout: TypeAlias = Mapping[str, Sequence[GPUSlot]]


@dataclass(frozen=True)
class ReplicaSlot:
    """One vLLM replica's permanent reservation of a contiguous GPU range.

    `slot_id` is assigned once by `build_slot_table` and is never reused, so it is
    safe to key long-lived state on. The slot fans out to `dp_size` of the
    `RolloutInstanceId`s that the coordinator and the PS track.
    """

    slot_id: int
    tp: int
    node_id: str
    start_offset: int
    """Offset into the node's pool GPUs, not a `gpu_id`. The pool need not start at GPU 0."""
    gpu_slots: frozenset[GPUSlot]

    @property
    def end_offset(self) -> int:
        """One past the last offset this slot covers, as a half-open bound."""
        return self.start_offset + self.tp

    def __repr__(self) -> str:
        span = f"[{self.start_offset},{self.end_offset})"
        return f"ReplicaSlot(id={self.slot_id}, tp={self.tp}, {self.node_id}:{span})"


class ReplicaSlotTable:
    """Every `ReplicaSlot` the rollout pool can run, built once at startup.

    GPU ranges are aligned, meaning a TP-4 slot covers GPUs 0-3 or 4-7 but never
    1-4. Two slots therefore either nest or stay disjoint, which makes
    `conflicts` an interval containment test rather than a set intersection, and
    keeps the slot count linear in the number of GPUs.
    """

    def __init__(self, slots: Sequence[ReplicaSlot]) -> None:
        self.slots: tuple[ReplicaSlot, ...] = tuple(slots)
        self.by_id: dict[int, ReplicaSlot] = {slot.slot_id: slot for slot in self.slots}
        grouped: dict[int, list[ReplicaSlot]] = defaultdict(list)
        for slot in self.slots:
            grouped[slot.tp].append(slot)
        self.by_tp: dict[int, tuple[ReplicaSlot, ...]] = {tp: tuple(group) for tp, group in grouped.items()}

    @property
    def tp_degrees(self) -> tuple[int, ...]:
        """Configured TP degrees, ascending."""
        return tuple(sorted(self.by_tp))

    @property
    def node_order(self) -> tuple[str, ...]:
        """Node ids in the order slots were assigned, which is sorted by node id.

        A slot's placement-group bundle index is its node's position here times the
        node's GPU count, plus the slot's own offset. Keeping the order on the table
        means callers never re-derive it.
        """
        seen: list[str] = []
        for slot in self.slots:
            if slot.node_id not in seen:
                seen.append(slot.node_id)
        return tuple(seen)

    def conflicts(self, a: ReplicaSlot, b: ReplicaSlot) -> bool:
        """Report whether two slots share a GPU and so cannot be awake together.

        Args:
            a (ReplicaSlot): One slot.
            b (ReplicaSlot): The other slot.

        Returns:
            bool: True if the two slots overlap on at least one GPU.
        """
        if a.slot_id == b.slot_id:
            return False
        if a.node_id != b.node_id:
            return False
        return a.start_offset < b.end_offset and b.start_offset < a.end_offset

    def gpu_count(self) -> int:
        """Return the number of distinct physical GPUs the pool spans."""
        return len({gpu for slot in self.slots for gpu in slot.gpu_slots})

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self) -> Iterator[ReplicaSlot]:
        return iter(self.slots)

    def __repr__(self) -> str:
        per_tp = ", ".join(f"tp{tp}x{len(self.by_tp[tp])}" for tp in self.tp_degrees)
        return f"ReplicaSlotTable({len(self.slots)} slots: {per_tp})"


def build_slot_table(
    node_to_gpu_slots: NodeGPULayout,
    tp_degrees: Sequence[int],
    pipeline_model_parallel_size: int = 1,
) -> ReplicaSlotTable:
    """Build the aligned slot table for a rollout pool.

    Each node is tiled independently for every TP degree, so a TP group never
    spans the inter-node fabric and `max(tp_degrees)` is capped by the smallest
    node's GPU count.

    Args:
        node_to_gpu_slots (NodeGPULayout): GPU slots each node contributes to the rollout pool.
        tp_degrees (Sequence[int]): TP degrees to materialize, such as (1, 2, 4, 8).
            Duplicates are ignored.
        pipeline_model_parallel_size (int): Must be 1. vLLM rejects PP > 1 in verl's
            rollout config, so a slot carries no PP dimension.

    Returns:
        ReplicaSlotTable: Slots ordered by node, then descending TP, then GPU offset.

    Raises:
        ValueError: If PP > 1, if `tp_degrees` is empty or non-positive, or if any
            TP degree does not evenly tile every node.
    """
    if pipeline_model_parallel_size != 1:
        raise ValueError(
            "dynamic_parallel supports pipeline_model_parallel_size=1 only, got "
            f"{pipeline_model_parallel_size!r}. vLLM raises NotImplementedError for PP > 1 in "
            "verl's rollout config, so a replica slot carries no PP dimension."
        )
    if not tp_degrees:
        raise ValueError("tp_degrees must name at least one TP degree, got an empty sequence.")
    if not node_to_gpu_slots:
        raise ValueError("node_to_gpu_slots must describe at least one node, got an empty mapping.")

    unique_degrees = sorted(set(tp_degrees))
    for tp in unique_degrees:
        if tp < 1:
            raise ValueError(f"TP degrees must be positive, got {tp!r}.")

    slots: list[ReplicaSlot] = []
    next_slot_id = 0
    # Descending TP within a node keeps the highest-TP slots at the front of the
    # table, which is also the order staged bootstrap must initialize them in.
    for node_id in sorted(node_to_gpu_slots):
        node_gpus = sorted(node_to_gpu_slots[node_id], key=lambda s: s.gpu_id)
        gpus_on_node = len(node_gpus)
        if gpus_on_node == 0:
            raise ValueError(f"Node {node_id!r} contributes no GPUs to the rollout pool.")
        for tp in sorted(unique_degrees, reverse=True):
            if tp > gpus_on_node:
                raise ValueError(
                    f"TP degree {tp} exceeds the {gpus_on_node} GPUs node {node_id!r} contributes. "
                    "A TP group must fit inside one node."
                )
            if gpus_on_node % tp != 0:
                raise ValueError(
                    f"TP degree {tp} does not evenly tile the {gpus_on_node} GPUs on node {node_id!r}. "
                    "Every TP degree must divide the per-node GPU count."
                )
            for start in range(0, gpus_on_node, tp):
                covered = node_gpus[start : start + tp]
                slots.append(
                    ReplicaSlot(
                        slot_id=next_slot_id,
                        tp=tp,
                        node_id=node_id,
                        start_offset=start,
                        gpu_slots=frozenset(covered),
                    )
                )
                next_slot_id += 1

    return ReplicaSlotTable(slots)
