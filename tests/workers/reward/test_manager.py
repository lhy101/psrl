"""CPU tests for the RewardModelManager interface."""

import inspect

import pytest
from psrl.workers.reward.reward_model.manager import RewardModelManager

pytestmark = pytest.mark.cpu_test


def test_manager_requires_gateway_url():
    """RewardModelManager must take the gateway URL, not the retired status queues."""
    sig = inspect.signature(RewardModelManager.__init__)
    assert "gateway_url" in sig.parameters, "RewardModelManager.__init__ must have a gateway_url parameter."
    assert "status_queues" not in sig.parameters, (
        "RewardModelManager.__init__ must not have status_queues, which is the retired API."
    )


def test_manager_has_get_gateway_url():
    """RewardModelManager must expose get_gateway_url()."""
    assert hasattr(RewardModelManager, "get_gateway_url")


def test_manager_has_get_reward_model_tokenizer():
    """RewardModelManager must expose get_reward_model_tokenizer()."""
    assert hasattr(RewardModelManager, "get_reward_model_tokenizer")
