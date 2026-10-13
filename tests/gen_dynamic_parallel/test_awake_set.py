"""CPU tests: `AwakeSet` validation and transition planning."""

import dataclasses

import pytest
from psrl.utils.gpu_topology import GPUSlot
from psrl.workers.gen.dynamic_parallel.awake_set import AwakeSet, SlotTransition
from psrl.workers.gen.dynamic_parallel.slots import build_slot_table

pytestmark = pytest.mark.cpu_test


def _table(n_gpus=8, tp_degrees=(1, 2, 4, 8), node_id="node1"):
    layout = {node_id: [GPUSlot(node_id=node_id, gpu_id=i) for i in range(n_gpus)]}
    return build_slot_table(layout, tp_degrees)


# --- Initial state ---


def test_starts_empty():
    aset = AwakeSet(_table())
    assert aset.current() == frozenset()
    assert aset.current_ids() == frozenset()


def test_empty_target_from_empty_state_is_a_noop():
    aset = AwakeSet(_table())
    assert aset.plan_transition([]).is_empty


# --- Validation ---


def test_rejects_conflicting_target():
    """Two slots sharing a GPU must never be awake together."""
    table = _table()
    aset = AwakeSet(table)
    tp8 = table.by_tp[8][0]
    tp1 = table.by_tp[1][0]
    with pytest.raises(ValueError, match="share a GPU"):
        aset.validate([tp8, tp1])


def test_accepts_disjoint_target_of_mixed_degrees():
    """A heterogeneous awake set is the whole point. TP-4 plus 4 TP-1 slots tiles 8 GPUs."""
    table = _table()
    aset = AwakeSet(table)
    tp4_left = next(s for s in table.by_tp[4] if s.start_offset == 0)
    tp1_right = [s for s in table.by_tp[1] if s.start_offset >= 4]
    target = aset.validate([tp4_left, *tp1_right])
    assert len(target) == 5


def test_accepts_full_tiling_of_single_degree():
    table = _table()
    aset = AwakeSet(table)
    assert len(aset.validate(table.by_tp[2])) == 4


def test_rejects_slot_id_beyond_the_table():
    table = _table(n_gpus=2, tp_degrees=(1, 2))
    aset = AwakeSet(table)
    big_table = _table()
    beyond = max(big_table.slots, key=lambda s: s.slot_id)
    with pytest.raises(KeyError, match="holds ids 0 to"):
        aset.validate([beyond])


def test_rejects_slot_from_a_different_layout():
    """`slot_id` is unique only within one table, so identity must be checked too."""
    table = _table()
    aset = AwakeSet(table)
    foreign = _table(n_gpus=4, tp_degrees=(1, 2, 4), node_id="other").by_tp[1][0]
    with pytest.raises(KeyError, match="came from a different layout"):
        aset.validate([foreign])


def test_rejects_stale_slot_object_reusing_a_known_id():
    """A mutated slot sharing an id with the table would silently alias another GPU range."""
    table = _table()
    aset = AwakeSet(table)
    tp1 = table.by_tp[1][0]
    impostor = dataclasses.replace(tp1, start_offset=7, gpu_slots=frozenset({GPUSlot("node1", 7)}))
    with pytest.raises(KeyError, match="came from a different layout"):
        aset.validate([impostor])


# --- Transition planning ---


def test_wake_from_empty_sleeps_nothing():
    table = _table()
    aset = AwakeSet(table)
    transition = aset.plan_transition(table.by_tp[8])
    assert transition.to_sleep == ()
    assert len(transition.to_wake) == 1
    assert transition.unchanged == ()


def test_transition_is_minimal():
    """A switch touching half the GPUs must leave the other half untouched."""
    table = _table()
    aset = AwakeSet(table)
    tp1 = sorted(table.by_tp[1], key=lambda s: s.start_offset)
    aset.commit(aset.plan_transition(tp1))

    # Replace the left 4 TP-1 slots with one TP-4 slot, keeping the right 4 as they are.
    tp4_left = next(s for s in table.by_tp[4] if s.start_offset == 0)
    transition = aset.plan_transition([tp4_left, *tp1[4:]])

    assert {s.slot_id for s in transition.to_sleep} == {s.slot_id for s in tp1[:4]}
    assert [s.slot_id for s in transition.to_wake] == [tp4_left.slot_id]
    assert {s.slot_id for s in transition.unchanged} == {s.slot_id for s in tp1[4:]}


def test_identical_target_is_a_noop_transition():
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[2]))
    transition = aset.plan_transition(table.by_tp[2])
    assert transition.is_empty
    assert len(transition.unchanged) == 4


def test_full_swap_sleeps_all_and_wakes_all():
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[1]))
    transition = aset.plan_transition(table.by_tp[8])
    assert len(transition.to_sleep) == 8
    assert len(transition.to_wake) == 1
    assert transition.unchanged == ()


def test_target_conflicting_with_itself_is_rejected_before_any_sleep():
    """Planning must fail without mutating state, so a bad plan cannot half-apply."""
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[8]))
    before = aset.current()
    with pytest.raises(ValueError):
        aset.plan_transition([table.by_tp[8][0], table.by_tp[1][0]])
    assert aset.current() == before


def test_plan_does_not_mutate_state():
    table = _table()
    aset = AwakeSet(table)
    aset.plan_transition(table.by_tp[4])
    assert aset.current() == frozenset()


def test_sleep_and_wake_sets_are_disjoint():
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[2]))
    transition = aset.plan_transition(table.by_tp[4])
    sleep_ids = {s.slot_id for s in transition.to_sleep}
    wake_ids = {s.slot_id for s in transition.to_wake}
    assert not (sleep_ids & wake_ids)


def test_transitions_are_ordered_by_slot_id():
    """Deterministic order keeps switch logs comparable across runs."""
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[1]))
    transition = aset.plan_transition(table.by_tp[4])
    assert [s.slot_id for s in transition.to_sleep] == sorted(s.slot_id for s in transition.to_sleep)
    assert [s.slot_id for s in transition.to_wake] == sorted(s.slot_id for s in transition.to_wake)


# --- Commit ---


def test_commit_applies_transition():
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[4]))
    assert aset.current_ids() == {s.slot_id for s in table.by_tp[4]}


def test_commit_is_idempotent():
    table = _table()
    aset = AwakeSet(table)
    transition = aset.plan_transition(table.by_tp[4])
    aset.commit(transition)
    first = aset.current()
    aset.commit(transition)
    assert aset.current() == first


def test_commit_rejects_transition_that_would_conflict():
    """Committing a plan built against a different state must fail, not corrupt the set."""
    table = _table()
    aset = AwakeSet(table)
    tp8 = table.by_tp[8][0]
    aset.commit(SlotTransition(to_sleep=(), to_wake=(tp8,), unchanged=()))
    with pytest.raises(ValueError, match="share a GPU"):
        aset.commit(SlotTransition(to_sleep=(), to_wake=(table.by_tp[1][0],), unchanged=()))


def test_sequence_of_switches_round_trips():
    table = _table()
    aset = AwakeSet(table)
    for degree in (1, 2, 4, 8, 1):
        aset.commit(aset.plan_transition(table.by_tp[degree]))
        assert aset.current_ids() == {s.slot_id for s in table.by_tp[degree]}


def test_awake_set_is_always_disjoint_across_many_switches():
    """The core safety invariant, asserted after every committed switch."""
    table = _table()
    aset = AwakeSet(table)
    for degree in (8, 1, 4, 2, 8, 4):
        aset.commit(aset.plan_transition(table.by_tp[degree]))
        awake = sorted(aset.current(), key=lambda s: s.slot_id)
        for i, a in enumerate(awake):
            for b in awake[i + 1 :]:
                assert not table.conflicts(a, b)


def test_sleep_to_empty_releases_everything():
    table = _table()
    aset = AwakeSet(table)
    aset.commit(aset.plan_transition(table.by_tp[2]))
    transition = aset.plan_transition([])
    assert len(transition.to_sleep) == 4
    aset.commit(transition)
    assert aset.current() == frozenset()
