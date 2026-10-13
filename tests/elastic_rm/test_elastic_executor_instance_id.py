"""CPU tests: ElasticExecutor keys all of its state by RolloutInstanceId."""

from unittest.mock import MagicMock

import pytest
from psrl.utils.elastic_rm.cluster_topology import ClusterTopology, GPUSlot, InstanceStatus
from psrl.utils.elastic_rm.elastic_executor import ElasticExecutor

pytestmark = pytest.mark.cpu_test

ROLE = "rollout"
MODEL = "test_model"


class _FakePolicy:
    min_awake_per_role = 0

    def decide(self, *args, **kwargs):
        return MagicMock(actions=[])


def _make_executor():
    """Build an ElasticExecutor without running the Ray actor constructor."""
    # `ray.remote` wraps the class, so reach the plain class behind the handle.
    executor_cls = ElasticExecutor.__ray_metadata__.modified_class
    executor = executor_cls.__new__(executor_cls)
    executor.coordinators = {ROLE: {MODEL: MagicMock()}}
    executor.roles = [(ROLE, MODEL)]
    executor.instances_status_flags = {}
    executor.instances_engine_stats = {}
    executor.topology = ClusterTopology()
    executor.scaling_policy = _FakePolicy()
    executor.elastic_rm_config = {}
    return executor


def _slots(node_id, gpu_ids):
    return frozenset(GPUSlot(node_id=node_id, gpu_id=g) for g in gpu_ids)


def test_register_role_keys_state_by_instance_id_tuple():
    executor = _make_executor()
    ids = [("wid-0", 0), ("wid-0", 1), ("wid-1", 0)]
    gpu_slots = [_slots("node1", [0]), _slots("node1", [1]), _slots("node1", [2])]

    executor.register_role(ROLE, MODEL, ids, gpu_slots)

    for instance_id in ids:
        assert instance_id in executor.instances_status_flags[ROLE][MODEL]
        assert executor.instances_status_flags[ROLE][MODEL][instance_id] == InstanceStatus.ASLEEP


def test_register_role_rejects_mismatched_gpu_slot_count():
    """A gpu_slots list of the wrong length is a caller bug, not a silent truncation."""
    executor = _make_executor()
    with pytest.raises(ValueError):
        executor.register_role(ROLE, MODEL, [("wid-0", 0), ("wid-0", 1)], [_slots("node1", [0])])


def test_topology_records_gpu_placement_under_tuple_key():
    executor = _make_executor()
    executor.register_role(ROLE, MODEL, [("wid-0", 0)], [_slots("node1", [0, 1])])
    assert executor.topology.get_gpu_slots(ROLE, MODEL, ("wid-0", 0)) == _slots("node1", [0, 1])


def test_select_initial_awake_ids_skips_gpu_conflicts():
    """Two instances sharing a GPU must not both be selected to wake."""
    executor = _make_executor()
    ids = [("wid-0", 0), ("wid-1", 0), ("wid-2", 0)]
    # `wid-1` overlaps `wid-0` on GPU 0, so only one of them may wake.
    gpu_slots = [_slots("node1", [0]), _slots("node1", [0]), _slots("node1", [1])]
    executor.register_role(ROLE, MODEL, ids, gpu_slots)

    selected = executor.select_initial_awake_ids(ROLE, MODEL, target_awake_num=3)

    assert ("wid-0", 0) in selected
    assert ("wid-1", 0) not in selected
    assert ("wid-2", 0) in selected
    for instance_id in selected:
        assert executor.instances_status_flags[ROLE][MODEL][instance_id] == InstanceStatus.AWAKEN


def test_engine_stats_stored_under_tuple_key():
    executor = _make_executor()
    executor.register_role(ROLE, MODEL, [("wid-0", 0)], [_slots("node1", [0])])
    executor.instances_engine_stats[ROLE][MODEL][("wid-0", 0)] = {"scheduler_stats": {}}
    assert ("wid-0", 0) in executor.instances_engine_stats[ROLE][MODEL]
