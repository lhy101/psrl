"""CPU tests: `SwitchMixin` call ordering and failure handling.

The switch is the one part of `dynamic_parallel` that touches Ray and SMG, so these
tests drive it against a fake coordinator that records calls. The property under
test is ordering: every outgoing slot must be asleep before any incoming slot wakes,
because a GPU's pages are only free once the outgoing slot has released them.
"""

import pytest
from psrl.utils.gpu_topology import GPUSlot
from psrl.workers.gen.dynamic_parallel import AwakeSet, build_slot_table
from psrl.workers.gen.dynamic_parallel.switch import SwitchMixin

pytestmark = pytest.mark.cpu_test


def _table(n_gpus=8, tp_degrees=(1, 2, 4, 8), node_id="node1"):
    layout = {node_id: [GPUSlot(node_id=node_id, gpu_id=i) for i in range(n_gpus)]}
    return build_slot_table(layout, tp_degrees)


class FakeCoordinator(SwitchMixin):
    """Records the calls `switch_to` makes, in order."""

    def __init__(self, table, dp_size=1, fail_on_wake=False, fail_on_sync=False):
        self.table = table
        self.awake_set = AwakeSet(table)
        self.calls: list[tuple] = []
        self.dp_size = dp_size
        self.fail_on_wake = fail_on_wake
        self.fail_on_sync = fail_on_sync

        # Every slot has a registered replica, as staged bootstrap would leave it.
        self.slot_to_replica_id = {slot.slot_id: f"w{slot.slot_id}" for slot in table}
        self.instance_ids = {(f"w{slot.slot_id}", dp_rank) for slot in table for dp_rank in range(dp_size)}
        self.server_handles = {f"w{slot.slot_id}": object() for slot in table}

    async def _set_routing_loop_running(self, running):
        self.calls.append(("routing", running))

    async def _gateway_post_json(self, path, payload, params=None):
        self.calls.append(("gateway", path, len(payload) if payload else 0))
        return {}

    async def exec_command(self, command, blocking=True):
        self.calls.append(("abort", len(command.get_args().get("instance_ids", []))))
        return 0

    async def _do_sleep_instance(self, replica_id):
        self.calls.append(("sleep", replica_id))

    async def _do_wake_up_instance(self, replica_id):
        if self.fail_on_wake:
            raise RuntimeError("wake failed")
        self.calls.append(("wake", replica_id))

    async def sync_with_ps(self, instance_ids, wait_model_sync=False, **kwargs):
        if self.fail_on_sync:
            raise RuntimeError("sync failed")
        self.calls.append(("sync", len(instance_ids), wait_model_sync))

    async def _quarantine_failed_replicas(self, replica_ids, instance_ids):
        self.calls.append(("quarantine", tuple(sorted(replica_ids))))

    def kinds(self):
        return [c[0] for c in self.calls]


async def _wake_degree(coord, tp):
    return await coord.switch_to(frozenset(coord.table.by_tp[tp]))


# --- Happy path ---


async def test_first_switch_wakes_without_sleeping():
    table = _table()
    coord = FakeCoordinator(table)
    transition = await _wake_degree(coord, 8)

    assert transition is not None
    assert "sleep" not in coord.kinds()
    assert ("wake", "w0") in coord.calls
    assert coord.awake_set.current_ids() == {0}


async def test_noop_switch_touches_nothing():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 8)
    coord.calls.clear()

    assert await _wake_degree(coord, 8) is None
    assert coord.calls == []


async def test_sleep_strictly_precedes_wake():
    """The core ordering invariant. A GPU's pages must be freed before remapping."""
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    coord.calls.clear()

    await _wake_degree(coord, 8)

    kinds = coord.kinds()
    assert kinds.index("wake") > max(i for i, k in enumerate(kinds) if k == "sleep")


async def test_all_outgoing_slots_sleep():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    coord.calls.clear()

    await _wake_degree(coord, 8)

    slept = {c[1] for c in coord.calls if c[0] == "sleep"}
    assert slept == {f"w{slot.slot_id}" for slot in table.by_tp[1]}


async def test_partial_switch_leaves_untouched_slots_alone():
    """A switch over half the GPUs must not sleep or wake the other half."""
    table = _table()
    coord = FakeCoordinator(table)
    tp1 = sorted(table.by_tp[1], key=lambda s: s.start_offset)
    await coord.switch_to(frozenset(tp1))
    coord.calls.clear()

    tp4_left = next(s for s in table.by_tp[4] if s.start_offset == 0)
    await coord.switch_to(frozenset({tp4_left, *tp1[4:]}))

    touched = {c[1] for c in coord.calls if c[0] in ("sleep", "wake")}
    kept = {f"w{s.slot_id}" for s in tp1[4:]}
    assert not (touched & kept)
    assert {c[1] for c in coord.calls if c[0] == "sleep"} == {f"w{s.slot_id}" for s in tp1[:4]}


async def test_routing_pauses_before_sleep_and_resumes_after():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    coord.calls.clear()
    await _wake_degree(coord, 8)

    routing = [(i, c[1]) for i, c in enumerate(coord.calls) if c[0] == "routing"]
    assert routing[0][1] is False
    assert routing[-1][1] is True
    first_sleep = min(i for i, k in enumerate(coord.kinds()) if k == "sleep")
    assert routing[0][0] < first_sleep


async def test_weights_are_pulled_after_wake():
    """A level-2 wake leaves weights unmapped, so the pull is mandatory."""
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 4)

    kinds = coord.kinds()
    assert "sync" in kinds
    assert kinds.index("sync") > max(i for i, k in enumerate(kinds) if k == "wake")
    sync_call = next(c for c in coord.calls if c[0] == "sync")
    assert sync_call[2] is True, "The switch must not return before weights are in place."


async def test_sync_covers_every_woken_instance():
    table = _table()
    coord = FakeCoordinator(table, dp_size=2)
    await _wake_degree(coord, 2)

    sync_call = next(c for c in coord.calls if c[0] == "sync")
    # 4 TP-2 slots x 2 dp ranks.
    assert sync_call[1] == 8


async def test_in_flight_requests_are_aborted_on_outgoing_slots():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    coord.calls.clear()
    await _wake_degree(coord, 8)

    abort = next(c for c in coord.calls if c[0] == "abort")
    assert abort[1] == 8, "Every outgoing instance should be aborted."
    kinds = coord.kinds()
    assert kinds.index("abort") < kinds.index("sleep")


async def test_abort_can_be_disabled():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    coord.calls.clear()
    await coord.switch_to(frozenset(table.by_tp[8]), abort_in_flight=False)

    assert "abort" not in coord.kinds()


async def test_switch_cycle_round_trips():
    table = _table()
    coord = FakeCoordinator(table)
    for tp in (1, 2, 4, 8, 1):
        await _wake_degree(coord, tp)
        assert coord.awake_set.current_ids() == {s.slot_id for s in table.by_tp[tp]}


# --- Failure handling ---


async def test_failed_wake_leaves_awake_set_uncommitted():
    """A half-applied switch must not be recorded, so the next boundary replans."""
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 1)
    before = coord.awake_set.current()

    coord.fail_on_wake = True
    with pytest.raises(RuntimeError, match="wake failed"):
        await _wake_degree(coord, 8)

    assert coord.awake_set.current() == before
    assert "quarantine" in coord.kinds()


async def test_failed_wake_resumes_routing():
    """Routing must not stay paused after a failed switch, or the job wedges."""
    table = _table()
    coord = FakeCoordinator(table, fail_on_wake=True)
    with pytest.raises(RuntimeError):
        await _wake_degree(coord, 8)

    routing = [c[1] for c in coord.calls if c[0] == "routing"]
    assert routing[-1] is True


async def test_failed_sync_propagates():
    table = _table()
    coord = FakeCoordinator(table, fail_on_sync=True)
    with pytest.raises(RuntimeError, match="sync failed"):
        await _wake_degree(coord, 8)
    assert coord.awake_set.current() == frozenset()


async def test_conflicting_target_rejected_before_any_call():
    table = _table()
    coord = FakeCoordinator(table)
    await _wake_degree(coord, 8)
    coord.calls.clear()

    with pytest.raises(ValueError, match="share a GPU"):
        await coord.switch_to(frozenset({table.by_tp[8][0], table.by_tp[1][0]}))

    assert coord.calls == [], "A rejected plan must not touch any engine."


async def test_unregistered_slot_is_skipped_not_fatal():
    """A slot without a replica id is logged and skipped rather than crashing."""
    table = _table()
    coord = FakeCoordinator(table)
    tp8 = table.by_tp[8][0]
    del coord.slot_to_replica_id[tp8.slot_id]

    transition = await coord.switch_to(frozenset({tp8}))

    assert transition is not None
    assert "wake" not in coord.kinds()
    # The awake set still records intent, so a later boundary can correct it.
    assert coord.awake_set.current_ids() == {tp8.slot_id}
