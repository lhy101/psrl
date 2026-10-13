"""CPU tests for RewardModelCoordinator.

A reward model has no parameter server, so the coordinator inherits the rollout
coordinator's machinery but neutralizes its weight-sync hooks.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from psrl.workers.gen.rollout_coordination import RolloutCoordinator
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator

pytestmark = pytest.mark.cpu_test


def _coordinator():
    """Build a coordinator without running the Ray actor constructor."""
    coord = object.__new__(RewardModelCoordinator)
    coord.config = MagicMock()
    coord.config.psrl.logging_path = "/tmp"
    coord.rm_config = MagicMock()
    coord.rm_config.reward_model_name = "TestRM"
    coord.reward_model_name = "TestRM"
    return coord


def test_coordinator_is_subclass():
    """RewardModelCoordinator must reuse the real RolloutCoordinator."""
    assert issubclass(RewardModelCoordinator, RolloutCoordinator)


def test_sync_model_is_noop():
    """There is no parameter server for a reward model, so sync must do nothing."""
    assert asyncio.run(_coordinator().sync_model()) is None


def test_update_model_version_is_noop():
    """A reward model is never versioned, so a version bump must do nothing."""
    assert asyncio.run(_coordinator().update_model_version(model_version=5)) is None


def test_sleep_level_retains_weights():
    """Level 1 frees KV cache only. A reward model cannot re-pull weights from a PS."""
    assert _coordinator()._get_sleep_level() == 1


def test_rollout_coordinator_sleep_level_releases_everything():
    """Contrast with rollout, which sleeps at level 2 because it can re-pull weights."""
    assert RolloutCoordinator._get_sleep_level(object.__new__(RolloutCoordinator)) == 2
