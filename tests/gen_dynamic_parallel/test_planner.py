"""CPU tests: `ParallelismPlan` invariants, the planner registry, and the static planner."""

import pytest
from omegaconf import OmegaConf
from psrl.utils.gpu_topology import GPUSlot
from psrl.workers.gen.dynamic_parallel.awake_set import AwakeSet
from psrl.workers.gen.dynamic_parallel.planner import (
    COHORT_NEW,
    COHORT_OLD,
    BoundaryState,
    CohortPlanner,
    ParallelismPlan,
    ScriptedCohortPlanner,
    StaticCohortPlanner,
    build_cohort_planner,
    get_cohort_planner_cls,
    register_cohort_planner,
)
from psrl.workers.gen.dynamic_parallel.slots import build_slot_table

pytestmark = pytest.mark.cpu_test


def _table(n_gpus=8, tp_degrees=(1, 2, 4, 8), node_id="node1"):
    layout = {node_id: [GPUSlot(node_id=node_id, gpu_id=i) for i in range(n_gpus)]}
    return build_slot_table(layout, tp_degrees)


def _state(model_version=0, staleness_budget=1):
    return BoundaryState(model_version=model_version, staleness_budget=staleness_budget)


# --- ParallelismPlan invariants ---


def test_plan_requires_a_label_for_every_awake_slot():
    """An unlabelled awake slot would leave request steering undefined."""
    table = _table()
    slots = table.by_tp[4]
    with pytest.raises(ValueError, match="missing labels"):
        ParallelismPlan(awake=frozenset(slots), cohort_of={slots[0].slot_id: COHORT_NEW})


def test_plan_rejects_labels_for_slots_that_are_not_awake():
    table = _table()
    awake = table.by_tp[4][0]
    asleep = table.by_tp[4][1]
    with pytest.raises(ValueError, match="not in the awake set"):
        ParallelismPlan(
            awake=frozenset({awake}),
            cohort_of={awake.slot_id: COHORT_NEW, asleep.slot_id: COHORT_OLD},
        )


def test_empty_plan_is_valid():
    assert ParallelismPlan(awake=frozenset(), cohort_of={}).awake == frozenset()


def test_slots_for_cohort_partitions_the_awake_set():
    table = _table()
    tp4_left = next(s for s in table.by_tp[4] if s.start_offset == 0)
    tp1_right = [s for s in table.by_tp[1] if s.start_offset >= 4]
    plan = ParallelismPlan(
        awake=frozenset({tp4_left, *tp1_right}),
        cohort_of={tp4_left.slot_id: COHORT_OLD, **{s.slot_id: COHORT_NEW for s in tp1_right}},
    )

    assert plan.slots_for_cohort(COHORT_OLD) == frozenset({tp4_left})
    assert plan.slots_for_cohort(COHORT_NEW) == frozenset(tp1_right)
    assert plan.slots_for_cohort(COHORT_OLD) | plan.slots_for_cohort(COHORT_NEW) == plan.awake


# --- Static planner ---


def test_static_planner_wakes_every_slot_of_its_degree():
    table = _table()
    planner = StaticCohortPlanner(tp_schedule=[2])
    plan = planner.plan(table, _state())
    assert plan.awake == frozenset(table.by_tp[2])
    assert len(plan.awake) == 4


def test_static_planner_single_degree_is_version_invariant():
    """A one-entry schedule is the no-switching baseline."""
    table = _table()
    planner = StaticCohortPlanner(tp_schedule=[4])
    first = planner.plan(table, _state(model_version=0))
    later = planner.plan(table, _state(model_version=7))
    assert first.awake == later.awake


def test_static_planner_cycles_a_multi_entry_schedule():
    """A multi-entry schedule drives the switch path without needing a solver."""
    table = _table()
    planner = StaticCohortPlanner(tp_schedule=[1, 8])
    assert planner.plan(table, _state(model_version=0)).awake == frozenset(table.by_tp[1])
    assert planner.plan(table, _state(model_version=1)).awake == frozenset(table.by_tp[8])
    assert planner.plan(table, _state(model_version=2)).awake == frozenset(table.by_tp[1])


def test_static_planner_output_is_always_gpu_disjoint():
    """Every plan must survive AwakeSet validation, for every degree in the table."""
    table = _table()
    for tp in table.tp_degrees:
        planner = StaticCohortPlanner(tp_schedule=[tp])
        plan = planner.plan(table, _state())
        AwakeSet(table).validate(plan.awake)


def test_static_planner_rejects_degree_absent_from_table():
    table = _table(n_gpus=4, tp_degrees=(1, 2))
    planner = StaticCohortPlanner(tp_schedule=[8])
    with pytest.raises(ValueError, match="not in the replica slot table"):
        planner.plan(table, _state())


def test_static_planner_rejects_empty_schedule():
    with pytest.raises(ValueError, match="at least one TP degree"):
        StaticCohortPlanner(tp_schedule=[])


def test_static_planner_rejects_non_positive_degree():
    with pytest.raises(ValueError, match="must be positive"):
        StaticCohortPlanner(tp_schedule=[2, 0])


def test_static_plan_is_applicable_by_awake_set():
    """The planner and the mechanism must compose without an adapter."""
    table = _table()
    aset = AwakeSet(table)
    planner = StaticCohortPlanner(tp_schedule=[1, 4])

    for version in range(4):
        plan = planner.plan(table, _state(model_version=version))
        aset.commit(aset.plan_transition(plan.awake))
        assert aset.current() == plan.awake


# --- Scripted planner ---


def _small_table():
    """4-GPU pool with TP-1 and TP-2: slot 0 = TP2 GPUs 0-1, slot 1 = TP2 GPUs 2-3,
    slots 2-5 = TP1 one GPU each."""
    from psrl.utils.gpu_topology import GPUSlot

    layout = {"n0": [GPUSlot(node_id="n0", gpu_id=i) for i in range(4)]}
    return build_slot_table(layout, tp_degrees=[1, 2])


def test_scripted_planner_wakes_named_slots():
    table = _small_table()
    planner = ScriptedCohortPlanner(slot_schedule=[[0, 4, 5]])
    plan = planner.plan(table, _state(model_version=0))
    assert {s.slot_id for s in plan.awake} == {0, 4, 5}


def test_scripted_planner_mixes_tp_degrees():
    table = _small_table()
    planner = ScriptedCohortPlanner(slot_schedule=[[0, 4, 5]])
    plan = planner.plan(table, _state(model_version=0))
    tps = {s.tp for s in plan.awake}
    assert tps == {1, 2}


def test_scripted_planner_cycles_schedule():
    table = _small_table()
    planner = ScriptedCohortPlanner(slot_schedule=[[0, 4, 5], [2, 3, 4, 5]])
    assert {s.slot_id for s in planner.plan(table, _state(0)).awake} == {0, 4, 5}
    assert {s.slot_id for s in planner.plan(table, _state(1)).awake} == {2, 3, 4, 5}
    assert {s.slot_id for s in planner.plan(table, _state(2)).awake} == {0, 4, 5}


def test_scripted_plan_is_gpu_disjoint():
    """Both mixed and homogeneous entries must pass AwakeSet validation."""
    table = _small_table()
    planner = ScriptedCohortPlanner(slot_schedule=[[0, 4, 5], [2, 3, 4, 5]])
    aset = AwakeSet(table)
    for version in range(4):
        plan = planner.plan(table, _state(model_version=version))
        aset.validate(plan.awake)


def test_scripted_plan_is_applicable_by_awake_set():
    """Scripted planner composes with AwakeSet commit correctly."""
    table = _small_table()
    aset = AwakeSet(table)
    planner = ScriptedCohortPlanner(slot_schedule=[[0, 4, 5], [2, 3, 4, 5]])

    for version in range(4):
        plan = planner.plan(table, _state(model_version=version))
        aset.commit(aset.plan_transition(plan.awake))
        assert aset.current() == plan.awake


def test_scripted_planner_rejects_unknown_slot_id():
    table = _small_table()
    planner = ScriptedCohortPlanner(slot_schedule=[[99]])
    with pytest.raises(KeyError, match="not in the replica slot table"):
        planner.plan(table, _state())


def test_scripted_planner_rejects_empty_schedule():
    with pytest.raises(ValueError, match="at least one entry"):
        ScriptedCohortPlanner(slot_schedule=[])


def test_scripted_planner_rejects_empty_entry():
    with pytest.raises(ValueError, match="at least one slot"):
        ScriptedCohortPlanner(slot_schedule=[[]])


def test_scripted_planner_rejects_duplicate_slots_in_entry():
    with pytest.raises(ValueError, match="same slot twice"):
        ScriptedCohortPlanner(slot_schedule=[[0, 0]])


def test_scripted_from_config_reads_slot_schedule():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"planner": "scripted", "slot_schedule": [[0, 4, 5], [2, 3, 4, 5]]})
    planner = ScriptedCohortPlanner.from_config(cfg)
    assert repr(planner) == "ScriptedCohortPlanner(slot_schedule=[[0, 4, 5], [2, 3, 4, 5]])"


def test_scripted_from_config_missing_slot_schedule_raises():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"planner": "scripted", "slot_schedule": None})
    with pytest.raises(ValueError, match="slot_schedule"):
        ScriptedCohortPlanner.from_config(cfg)


def test_build_cohort_planner_scripted():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"planner": "scripted", "slot_schedule": [[0], [1]]})
    planner = build_cohort_planner(cfg)
    assert isinstance(planner, ScriptedCohortPlanner)


# --- Registry ---


def test_registry_resolves_the_static_planner():
    assert get_cohort_planner_cls("static") is StaticCohortPlanner


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown cohort planner"):
        get_cohort_planner_cls("no_such_planner")


def test_build_from_config_uses_tp_schedule():
    config = OmegaConf.create({"planner": "static", "tp_schedule": [2, 4]})
    planner = build_cohort_planner(config)
    assert isinstance(planner, StaticCohortPlanner)
    table = _table()
    assert planner.plan(table, _state(model_version=0)).awake == frozenset(table.by_tp[2])
    assert planner.plan(table, _state(model_version=1)).awake == frozenset(table.by_tp[4])


def test_build_from_config_falls_back_to_first_tp_degree():
    """Without a schedule the planner holds the first configured degree, so nothing switches."""
    config = OmegaConf.create({"planner": "static", "tp_degrees": [2, 4, 8]})
    planner = build_cohort_planner(config)
    table = _table()
    assert planner.plan(table, _state(model_version=3)).awake == frozenset(table.by_tp[2])


def test_build_defaults_to_static_planner():
    config = OmegaConf.create({"tp_degrees": [1]})
    assert isinstance(build_cohort_planner(config), StaticCohortPlanner)


def test_registering_a_duplicate_name_with_a_different_class_raises():
    with pytest.raises(ValueError, match="already registered"):

        @register_cohort_planner("static")
        class _Impostor(CohortPlanner):
            @classmethod
            def from_config(cls, config):
                return cls()

            def plan(self, table, state):
                return ParallelismPlan(awake=frozenset(), cohort_of={})


def test_registering_the_same_class_twice_is_idempotent():
    @register_cohort_planner("test_idempotent")
    class _Planner(CohortPlanner):
        @classmethod
        def from_config(cls, config):
            return cls()

        def plan(self, table, state):
            return ParallelismPlan(awake=frozenset(), cohort_of={})

    register_cohort_planner("test_idempotent")(_Planner)
    assert get_cohort_planner_cls("test_idempotent") is _Planner


def test_custom_planner_is_reachable_through_build():
    @register_cohort_planner("test_highest_tp")
    class _HighestTP(CohortPlanner):
        @classmethod
        def from_config(cls, config):
            return cls()

        def plan(self, table, state):
            slots = table.by_tp[max(table.tp_degrees)]
            return ParallelismPlan(
                awake=frozenset(slots),
                cohort_of={s.slot_id: COHORT_OLD for s in slots},
            )

    config = OmegaConf.create({"planner": "test_highest_tp"})
    planner = build_cohort_planner(config)
    table = _table()
    assert planner.plan(table, _state()).awake == frozenset(table.by_tp[8])


def test_cohort_planner_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CohortPlanner()
