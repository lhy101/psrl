"""CPU test: ppo_trainer composes the PSRL reward group correctly."""

import os

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def reward_config():
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="ppo_trainer").reward


def test_reward_group_exposes_managers_and_active_managers(reward_config):
    assert isinstance(dict(reward_config.managers), dict)
    assert isinstance(list(reward_config.active_managers), list)


def test_yaml_turns_on_async_reward_launch(reward_config):
    """The dataclass defaults to False. The shipped YAML opts in, so scoring overlaps rollout."""
    assert reward_config.launch_reward_fn_async is True


def test_default_run_activates_only_the_naive_manager(reward_config):
    """A recipe opts into anything heavier by overriding active_managers."""
    assert list(reward_config.active_managers) == ["naive"]


def test_every_active_manager_is_declared(reward_config):
    """An active name with no declaration would fail at resolve time, not here."""
    undeclared = set(reward_config.active_managers) - set(reward_config.managers)
    assert not undeclared, f"active_managers names undeclared managers: {sorted(undeclared)}."


def test_dapo_manager_is_declared_for_recipes_to_activate(reward_config):
    assert "dapo" in reward_config.managers


def test_dapo_manager_structure(reward_config):
    dapo_config = reward_config.managers.dapo
    assert dapo_config.reward_loop_type == "dapo"
    assert dapo_config.reward_manager.name == "dapo"
    assert dapo_config.reward_manager.source == "register"
    assert dapo_config.reward_kwargs.max_resp_len == 5120
    assert dapo_config.reward_kwargs.overlong_buffer_cfg.enable is True
