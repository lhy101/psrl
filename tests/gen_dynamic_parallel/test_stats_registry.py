"""CPU tests: the stats instance registry that labels each stats file.

A filename like `stats_r3_dp0.jsonl` cannot say whether replica 3 serves rollout
or validation, nor what TP degree it runs. The registry carries that, and these
tests pin the shape analysis tools depend on.
"""

import json

import pytest
from psrl.workers.gen.rollout_coordination.coordinator import RolloutCoordinator
from psrl.workers.gen.stats_recorder import StatsRecorder

pytestmark = pytest.mark.cpu_test


class _Replica:
    def __init__(self, replica_rank, tp, dp=1):
        self.replica_rank = replica_rank
        self.tensor_parallel_size = tp
        self.data_parallel_size = dp


def _coordinator(rollout_tps, validate_count=0, slot_ids=None, dp=1):
    """Build a coordinator carrying only the fields the registry reads."""
    coordinator = RolloutCoordinator.__new__(RolloutCoordinator)
    replicas = {}
    for idx, tp in enumerate(rollout_tps):
        replicas[str(idx)] = _Replica(idx, tp, dp)
    offset = len(rollout_tps)
    for i in range(validate_count):
        replicas[str(offset + i)] = _Replica(offset + i, 1, dp)
    coordinator.rollout_replicas = replicas
    coordinator.tag_to_replica_ids = {
        "rollout": {str(i) for i in range(len(rollout_tps))},
        "validate": {str(offset + i) for i in range(validate_count)},
    }
    coordinator.slot_to_replica_id = slot_ids if slot_ids is not None else {}
    coordinator.awake_set = None
    return coordinator


def test_registry_labels_carry_role_and_tp():
    coordinator = _coordinator([2, 2, 1, 1], slot_ids={i: str(i) for i in range(4)})
    by_file = {e["stats_file"]: e for e in coordinator.build_stats_instance_registry()}

    assert by_file["stats_r0_dp0.jsonl"]["label"] == "rollout0 TP2 slot0"
    assert by_file["stats_r2_dp0.jsonl"]["label"] == "rollout2 TP1 slot2"


def test_registry_numbers_each_tag_independently():
    """Validate replicas restart at 0 rather than continuing the replica_rank."""
    coordinator = _coordinator([1, 1], validate_count=2)
    entries = coordinator.build_stats_instance_registry()
    validate = [e for e in entries if e["tag"] == "validate"]

    assert [e["tag_index"] for e in validate] == [0, 1]
    assert [e["label"] for e in validate] == ["validate0 TP1", "validate1 TP1"]


def test_registry_omits_slot_for_validate():
    coordinator = _coordinator([1], validate_count=1, slot_ids={0: "0"})
    by_file = {e["stats_file"]: e for e in coordinator.build_stats_instance_registry()}

    assert by_file["stats_r1_dp0.jsonl"]["slot_id"] is None
    assert "slot" not in by_file["stats_r1_dp0.jsonl"]["label"]


def test_registry_covers_mixed_tp_degrees():
    """A mixed awake set is the case the filename is least able to describe."""
    coordinator = _coordinator([2, 2, 1, 1, 1, 1], slot_ids={i: str(i) for i in range(6)})
    entries = coordinator.build_stats_instance_registry()

    assert {e["tp"] for e in entries} == {1, 2}
    assert len(entries) == 6


def test_registry_expands_data_parallel_ranks():
    coordinator = _coordinator([1], dp=3, slot_ids={0: "0"})
    entries = coordinator.build_stats_instance_registry()

    assert [e["stats_file"] for e in entries] == [
        "stats_r0_dp0.jsonl",
        "stats_r0_dp1.jsonl",
        "stats_r0_dp2.jsonl",
    ]


def test_registry_without_slots_still_labels():
    """Runs without dynamic parallelism have no slot ids and must still label."""
    coordinator = _coordinator([1, 1])
    entries = coordinator.build_stats_instance_registry()

    assert all(e["slot_id"] is None for e in entries)
    assert entries[0]["label"] == "rollout0 TP1"


def test_write_instance_registry_merges_into_existing_config(tmp_path):
    recorder = StatsRecorder.__new__(StatsRecorder)
    recorder._logging_path = str(tmp_path)
    config_path = tmp_path / "stats_config.json"
    config_path.write_text(json.dumps({"routing_strategy": "request_num_balance"}))

    recorder.write_instance_registry([{"stats_file": "stats_r0_dp0.jsonl", "label": "rollout0 TP1"}])

    payload = json.loads(config_path.read_text())
    assert payload["routing_strategy"] == "request_num_balance"
    assert payload["instances"][0]["label"] == "rollout0 TP1"


def test_write_instance_registry_creates_config_when_absent(tmp_path):
    recorder = StatsRecorder.__new__(StatsRecorder)
    recorder._logging_path = str(tmp_path)

    recorder.write_instance_registry([{"stats_file": "stats_r0_dp0.jsonl", "label": "rollout0 TP2"}])

    payload = json.loads((tmp_path / "stats_config.json").read_text())
    assert payload["instances"][0]["label"] == "rollout0 TP2"
