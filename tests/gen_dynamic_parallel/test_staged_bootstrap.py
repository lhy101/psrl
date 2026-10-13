"""CPU tests: staged-bootstrap wiring in `PSRL_RayPPOTrainer`.

Milestone 3 derives the slot count, the per-slot parallelism, and each slot's
placement-group bundle range from the slot table. Those are pure index
computations, and an off-by-one in any of them silently puts an engine on the
wrong GPU, so they are tested here without Ray.
"""

import pytest
from psrl.utils.gpu_topology import GPUSlot
from psrl.workers.gen.dynamic_parallel import build_slot_table

pytestmark = pytest.mark.cpu_test


def _layout(nnodes, gpus_per_node):
    return {f"node{n}": [GPUSlot(node_id=f"node{n}", gpu_id=g) for g in range(gpus_per_node)] for n in range(nnodes)}


def _bundle_range(table, slot, store):
    """Mirror of `_build_slot_resource_pool`'s index math in ray_trainer."""
    node_offset = table.node_order.index(slot.node_id)
    gpus_per_node = store[node_offset]
    start = node_offset * gpus_per_node + slot.start_offset
    return start, start + slot.tp


# --- Slot count drives n_rollout_instances ---


def test_slot_count_matches_engines_per_gpu():
    """n_rollout_instances becomes the slot count, so one worker group exists per slot."""
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    assert len(table) == 15
    # Each GPU hosts one engine per TP degree.
    assert len(table.tp_degrees) == 4


def test_slot_count_scales_with_nodes():
    table = build_slot_table(_layout(2, 4), (1, 2, 4))
    assert len(table) == 14
    assert table.node_order == ("node0", "node1")


# --- Per-slot parallelism ---


def test_every_slot_reports_its_own_tp():
    """`_resolve_rollout_parallel_config` reads tp off the slot, so slots must carry it."""
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    by_index = [table.slots[i].tp for i in range(len(table))]
    assert sorted(by_index, reverse=True) == by_index, "Descending TP order is assumed by bootstrap."
    assert set(by_index) == {1, 2, 4, 8}


def test_slot_world_size_equals_tp():
    """A slot maps to exactly one replica, which requires world_size == tp with dp=pp=1."""
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    for slot in table:
        assert len(slot.gpu_slots) == slot.tp


# --- Bundle range carving ---


def test_bundle_ranges_are_contiguous_and_in_pool():
    """SubRayResourcePool requires a contiguous range inside the pool."""
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    store = [8]
    for slot in table:
        start, end = _bundle_range(table, slot, store)
        assert 0 <= start < end <= sum(store)
        assert end - start == slot.tp


def test_bundle_ranges_match_gpu_ids_single_node():
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    store = [8]
    for slot in table:
        start, end = _bundle_range(table, slot, store)
        assert {g.gpu_id for g in slot.gpu_slots} == set(range(start, end))


def test_bundle_ranges_offset_by_node():
    """Node 1's bundles must start after node 0's, or slots would collide across nodes."""
    table = build_slot_table(_layout(2, 4), (1, 2, 4))
    store = [4, 4]
    node0 = [s for s in table if s.node_id == "node0"]
    node1 = [s for s in table if s.node_id == "node1"]
    for slot in node0:
        start, end = _bundle_range(table, slot, store)
        assert 0 <= start < end <= 4
    for slot in node1:
        start, end = _bundle_range(table, slot, store)
        assert 4 <= start < end <= 8


def test_same_tp_slots_tile_the_pool_without_overlap():
    table = build_slot_table(_layout(2, 4), (1, 2, 4))
    store = [4, 4]
    for tp in table.tp_degrees:
        covered = []
        for slot in table.by_tp[tp]:
            start, end = _bundle_range(table, slot, store)
            covered.extend(range(start, end))
        assert sorted(covered) == list(range(8)), f"TP {tp} must tile all 8 bundles exactly once."


def test_conflicting_slots_have_overlapping_bundle_ranges():
    """The conflict predicate and the bundle math must agree, or oversubscription breaks."""
    table = build_slot_table(_layout(1, 8), (1, 2, 4, 8))
    store = [8]
    for a in table:
        for b in table:
            a_start, a_end = _bundle_range(table, a, store)
            b_start, b_end = _bundle_range(table, b, store)
            overlaps = a_start < b_end and b_start < a_end and a.slot_id != b.slot_id
            assert table.conflicts(a, b) == overlaps


def test_node_order_is_stable():
    """Bundle offsets key on this order, so it must not depend on dict iteration order."""
    forward = build_slot_table(_layout(2, 4), (1, 2))
    reversed_layout = dict(reversed(list(_layout(2, 4).items())))
    backward = build_slot_table(reversed_layout, (1, 2))
    assert forward.node_order == backward.node_order == ("node0", "node1")
