"""CPU-only: `load_weight=False` lets an engine build a skeleton and pull weights later.

PSRL pulls weights from the parameter server over NIXL, so a train worker must be
able to build its module from config instead of from a checkpoint.
"""

import dataclasses

import pytest
from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig

pytestmark = pytest.mark.cpu_test

ENGINE_CONFIGS = [FSDPEngineConfig, McoreEngineConfig]


@pytest.mark.parametrize("config_cls", ENGINE_CONFIGS)
def test_load_weight_defaults_to_true(config_cls):
    """A plain run still loads a checkpoint, so the NIXL path must be opt-in."""
    assert config_cls().load_weight is True


@pytest.mark.parametrize("config_cls", ENGINE_CONFIGS)
def test_load_weight_can_be_set_false(config_cls):
    assert config_cls(load_weight=False).load_weight is False


@pytest.mark.parametrize("config_cls", ENGINE_CONFIGS)
def test_engine_config_is_frozen(config_cls):
    """Engine configs are frozen, so every field must be passed at construction."""
    config = config_cls(load_weight=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.load_weight = True


def test_fsdp_module_build_flags_are_constructor_arguments():
    """The flags `_build_module` reads must all be settable in one frozen construction."""
    config = FSDPEngineConfig(
        load_weight=False,
        strategy="fsdp",
        forward_only=False,
        entropy_from_logits_with_chunking=False,
        use_torch_compile=False,
    )

    assert config.load_weight is False
    assert config.strategy == "fsdp"
    assert config.forward_only is False
    assert config.entropy_from_logits_with_chunking is False
    assert config.use_torch_compile is False
