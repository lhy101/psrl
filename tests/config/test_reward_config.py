"""CPU tests for the reward config dataclasses.

Reward scoring is multi-manager: `MultiRewardModelConfig` holds every declared
manager, and `active_managers` names the subset a run actually uses.
"""

import dataclasses
from dataclasses import dataclass

import pytest
from psrl.workers.config import (
    MultiRewardModelConfig,
    RewardLoopWorkerConfig,
    SingleRewardModelConfig,
    resolve_active_managers,
)

pytestmark = pytest.mark.cpu_test


def test_multi_reward_model_config_defaults_to_no_managers():
    """A run without reward models must declare none, not a default one."""
    cfg = MultiRewardModelConfig()
    assert cfg.managers == {}
    assert cfg.active_managers == []
    assert cfg.launch_reward_fn_async is False


def test_single_reward_model_config_defaults():
    """Read defaults from the dataclass, since constructing one loads a tokenizer."""
    defaults = {f.name: f.default for f in dataclasses.fields(SingleRewardModelConfig)}
    assert defaults["reward_model_name"] is None
    assert defaults["reward_loop_type"] is None
    assert defaults["enable_resource_pool"] is False
    assert defaults["num_replicas"] == 1


def test_reward_loop_worker_config_defaults():
    cfg = RewardLoopWorkerConfig()
    assert cfg.enable is True
    assert cfg.num_workers == 8
    assert cfg.placement == "reward_service_ip"


def test_dataclasses_are_exported_from_workers_config():
    """These names are the public surface the trainer imports."""
    from psrl.workers.config import (  # noqa: F401
        MultiRewardModelConfig,
        RewardLoopWorkerConfig,
        RewardModelConfig,
        SingleRewardModelConfig,
    )


@dataclass
class _FakeManagerConfig:
    reward_loop_type: str = "naive"


@dataclass
class _FakeRewardConfig:
    active_managers: list = None
    managers: dict = None

    def __post_init__(self):
        if self.active_managers is None:
            self.active_managers = []
        if self.managers is None:
            self.managers = {}


def test_resolve_active_managers_preserves_declared_order():
    config = _FakeRewardConfig(
        active_managers=["dapo", "naive"],
        managers={"dapo": _FakeManagerConfig("dapo"), "naive": _FakeManagerConfig("naive")},
    )

    resolved = resolve_active_managers(config)

    assert [m.reward_loop_type for m in resolved] == ["dapo", "naive"]


def test_resolve_active_managers_ignores_declared_but_inactive():
    """A manager left out of active_managers must not be built."""
    config = _FakeRewardConfig(
        active_managers=["naive"],
        managers={"dapo": _FakeManagerConfig("dapo"), "naive": _FakeManagerConfig("naive")},
    )

    resolved = resolve_active_managers(config)

    assert [m.reward_loop_type for m in resolved] == ["naive"]


def test_resolve_active_managers_returns_empty_when_none_active():
    assert resolve_active_managers(_FakeRewardConfig()) == []


def test_resolve_active_managers_rejects_an_undeclared_name():
    """Naming a manager that was never declared is a config typo, not an empty list."""
    config = _FakeRewardConfig(active_managers=["nonexistent"], managers={})
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_active_managers(config)
