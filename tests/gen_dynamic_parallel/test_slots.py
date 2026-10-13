"""CPU tests: `ReplicaSlot` geometry and `build_slot_table` validation."""

import pytest
from psrl.utils.gpu_topology import GPUSlot
from psrl.workers.gen.dynamic_parallel.slots import build_slot_table

pytestmark = pytest.mark.cpu_test


def _node(node_id, n_gpus, first_gpu=0):
    return {node_id: [GPUSlot(node_id=node_id, gpu_id=first_gpu + i) for i in range(n_gpus)]}


# --- Slot counts ---


def test_slot_count_is_sum_of_g_over_tp():
    """8 GPUs with TP in {1,2,4,8} gives 8 + 4 + 2 + 1 = 15 slots."""
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    assert len(table) == 15
    assert len(table.by_tp[1]) == 8
    assert len(table.by_tp[2]) == 4
    assert len(table.by_tp[4]) == 2
    assert len(table.by_tp[8]) == 1


def test_slot_count_for_reduced_degree_set():
    """Narrowing the degree set is the documented fallback if memory is tight."""
    table = build_slot_table(_node("node1", 8), (1, 2))
    assert len(table) == 12


def test_single_tp_degree_matches_gpu_count():
    table = build_slot_table(_node("node1", 4), (1,))
    assert len(table) == 4
    assert table.gpu_count() == 4


def test_duplicate_tp_degrees_are_deduplicated():
    table = build_slot_table(_node("node1", 4), (1, 2, 2, 1))
    assert len(table) == 6
    assert table.tp_degrees == (1, 2)


def test_multi_node_tiles_each_node_independently():
    layout = {**_node("node1", 4), **_node("node2", 4)}
    table = build_slot_table(layout, (1, 2, 4))
    # Per node: 4 + 2 + 1 = 7 slots, so 14 across two nodes.
    assert len(table) == 14
    assert table.gpu_count() == 8
    for slot in table:
        node_ids = {gpu.node_id for gpu in slot.gpu_slots}
        assert node_ids == {slot.node_id}, "A TP group must not span nodes."


# --- Conflict matrix ---


def test_nested_slots_conflict():
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    tp8 = table.by_tp[8][0]
    for slot in table:
        if slot.slot_id != tp8.slot_id:
            assert table.conflicts(tp8, slot), "Every slot nests inside the TP-8 slot."


def test_sibling_slots_of_same_degree_are_disjoint():
    table = build_slot_table(_node("node1", 8), (4,))
    left, right = table.by_tp[4]
    assert not table.conflicts(left, right)


def test_slot_does_not_conflict_with_itself():
    """Self-conflict would make any awake set invalid."""
    table = build_slot_table(_node("node1", 4), (1, 2, 4))
    for slot in table:
        assert not table.conflicts(slot, slot)


def test_conflict_is_symmetric():
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    for a in table:
        for b in table:
            assert table.conflicts(a, b) == table.conflicts(b, a)


def test_conflict_matches_gpu_set_intersection():
    """The interval test is an optimization. It must agree with the set-based truth."""
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    for a in table:
        for b in table:
            overlaps = bool(a.gpu_slots & b.gpu_slots) and a.slot_id != b.slot_id
            assert table.conflicts(a, b) == overlaps


def test_slots_on_different_nodes_never_conflict():
    layout = {**_node("node1", 4), **_node("node2", 4)}
    table = build_slot_table(layout, (1, 4))
    node1_slots = [s for s in table if s.node_id == "node1"]
    node2_slots = [s for s in table if s.node_id == "node2"]
    for a in node1_slots:
        for b in node2_slots:
            assert not table.conflicts(a, b)


def test_same_gpu_id_on_two_nodes_does_not_alias():
    """Offsets are per node, so GPU 0 on two nodes must stay two GPUs."""
    layout = {**_node("node1", 2), **_node("node2", 2)}
    table = build_slot_table(layout, (1,))
    assert table.gpu_count() == 4


def test_pool_not_starting_at_gpu_zero():
    """A rollout pool may hold a GPU suffix when the trainer owns the prefix."""
    table = build_slot_table(_node("node1", 4, first_gpu=4), (1, 2, 4))
    assert len(table) == 7
    assert table.gpu_count() == 4
    tp4 = table.by_tp[4][0]
    assert {g.gpu_id for g in tp4.gpu_slots} == {4, 5, 6, 7}
    # Offsets stay pool-relative while gpu_ids stay physical.
    assert tp4.start_offset == 0


# --- Slot identity ---


def test_slot_ids_are_dense_and_unique():
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    ids = sorted(slot.slot_id for slot in table)
    assert ids == list(range(len(table)))


def test_slot_id_stable_under_reordered_gpu_input():
    """Input order must not perturb identity, since slot_id keys long-lived state."""
    forward = [GPUSlot("node1", i) for i in range(4)]
    shuffled = [forward[2], forward[0], forward[3], forward[1]]

    a = build_slot_table({"node1": forward}, (1, 2, 4))
    b = build_slot_table({"node1": shuffled}, (1, 2, 4))

    assert [(s.slot_id, s.tp, s.start_offset) for s in a] == [(s.slot_id, s.tp, s.start_offset) for s in b]


def test_slot_id_stable_under_reordered_tp_degrees():
    a = build_slot_table(_node("node1", 4), (1, 2, 4))
    b = build_slot_table(_node("node1", 4), (4, 1, 2))
    assert [(s.slot_id, s.tp) for s in a] == [(s.slot_id, s.tp) for s in b]


def test_slot_id_stable_under_reordered_nodes():
    forward = {**_node("node1", 2), **_node("node2", 2)}
    reverse = {**_node("node2", 2), **_node("node1", 2)}
    a = build_slot_table(forward, (1, 2))
    b = build_slot_table(reverse, (1, 2))
    assert [(s.slot_id, s.node_id, s.tp) for s in a] == [(s.slot_id, s.node_id, s.tp) for s in b]


def test_highest_tp_comes_first_within_a_node():
    """Staged bootstrap initializes in descending TP, so the table is already in that order."""
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    tps = [slot.tp for slot in table]
    assert tps == sorted(tps, reverse=True)


def test_by_id_lookup():
    table = build_slot_table(_node("node1", 4), (1, 2))
    for slot in table:
        assert table.by_id[slot.slot_id] is slot


def test_gpu_ranges_tile_the_node_exactly():
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    for tp in table.tp_degrees:
        covered = [gpu for slot in table.by_tp[tp] for gpu in slot.gpu_slots]
        assert len(covered) == 8, f"TP {tp} must cover all 8 GPUs."
        assert len(set(covered)) == 8, f"TP {tp} must cover each GPU exactly once."


def test_slot_gpu_count_equals_tp():
    table = build_slot_table(_node("node1", 8), (1, 2, 4, 8))
    for slot in table:
        assert len(slot.gpu_slots) == slot.tp


# --- Validation ---


def test_rejects_pipeline_parallel():
    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        build_slot_table(_node("node1", 4), (1, 2), pipeline_model_parallel_size=2)


def test_rejects_tp_exceeding_node_gpu_count():
    with pytest.raises(ValueError, match="exceeds"):
        build_slot_table(_node("node1", 4), (1, 8))


def test_rejects_tp_not_dividing_node_gpu_count():
    with pytest.raises(ValueError, match="evenly tile"):
        build_slot_table(_node("node1", 6), (4,))


def test_rejects_empty_tp_degrees():
    with pytest.raises(ValueError, match="at least one TP degree"):
        build_slot_table(_node("node1", 4), ())


def test_rejects_empty_node_layout():
    with pytest.raises(ValueError, match="at least one node"):
        build_slot_table({}, (1,))


def test_rejects_node_with_no_gpus():
    with pytest.raises(ValueError, match="no GPUs"):
        build_slot_table({"node1": []}, (1,))


def test_rejects_non_positive_tp():
    with pytest.raises(ValueError, match="must be positive"):
        build_slot_table(_node("node1", 4), (0, 1))


def test_rejects_tp_valid_on_one_node_but_not_another():
    """A heterogeneous pool must fail loudly rather than tile only the node that fits."""
    layout = {**_node("node1", 4), **_node("node2", 2)}
    with pytest.raises(ValueError, match="exceeds"):
        build_slot_table(layout, (4,))
