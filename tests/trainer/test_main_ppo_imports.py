"""CPU-only: verify main_ppo wires the unified engine workers, not the deprecated ones."""

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


def _unwrap_ray_class(handle):
    """Return the class behind a `ray.remote` handle.

    `TaskRunner` is decorated at module scope, so the exported name is an
    `ActorClass` that cannot be instantiated outside Ray. Ray's wrapper is a
    *subclass* of the original, so callers compare with `issubclass` rather than
    identity.
    """
    return handle.__ray_metadata__.modified_class


def _assert_role_runs(handle, expected_cls):
    """Assert a registered Ray actor handle wraps `expected_cls`."""
    wrapped = _unwrap_ray_class(handle)
    assert issubclass(wrapped, expected_cls), f"Expected a {expected_cls.__name__} actor, got {wrapped!r}."


def _task_runner():
    """Build a TaskRunner without running its Ray actor machinery."""
    from psrl.trainer.main_ppo import TaskRunner

    runner_cls = _unwrap_ray_class(TaskRunner)
    runner = runner_cls.__new__(runner_cls)
    runner.role_worker_mapping = {}
    runner.mapping = {}
    return runner


def test_deprecated_fsdp_workers_not_imported_by_main_ppo():
    """main_ppo should not cause fsdp_workers/megatron_workers to be imported."""
    import sys

    # Snapshot modules before fresh import
    modules_before = set(sys.modules.keys())

    # Evict main_ppo so it re-executes on import
    for mod in list(sys.modules.keys()):
        if "psrl.trainer.main_ppo" in mod:
            del sys.modules[mod]

    import psrl.trainer.main_ppo  # noqa: F401

    modules_added = set(sys.modules.keys()) - modules_before

    assert not any("fsdp_workers" in m for m in modules_added), (
        f"main_ppo import pulled in fsdp_workers: {[m for m in modules_added if 'fsdp_workers' in m]}"
    )
    assert not any("megatron_workers" in m for m in modules_added), (
        f"main_ppo import pulled in megatron_workers: {[m for m in modules_added if 'megatron_workers' in m]}"
    )


def test_actor_and_rollout_roles_get_the_unified_workers():
    """The Actor role must map to the unified engine worker and Rollout to the vLLM adapter."""
    from psrl.trainer.ppo.utils import PSRL_Role
    from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    runner = _task_runner()
    config = OmegaConf.create({"psrl": {"colocate_validate_and_train": False}})
    runner.add_actor_rollout_worker(config)

    _assert_role_runs(runner.role_worker_mapping[PSRL_Role.Actor], PSRL_EngineTrainWorker)
    _assert_role_runs(runner.role_worker_mapping[PSRL_Role.Rollout], PSRL_ServerAdapter)
    assert runner.mapping[PSRL_Role.Actor] == ["train_pool"]


def test_validate_role_registered_only_when_colocated():
    """The Validate role shares train GPUs, so it exists only in colocated mode."""
    from psrl.trainer.ppo.utils import PSRL_Role

    runner = _task_runner()
    runner.add_actor_rollout_worker(OmegaConf.create({"psrl": {"colocate_validate_and_train": False}}))
    assert PSRL_Role.Validate not in runner.role_worker_mapping

    colocated = _task_runner()
    colocated.add_actor_rollout_worker(OmegaConf.create({"psrl": {"colocate_validate_and_train": True}}))
    assert PSRL_Role.Validate in colocated.role_worker_mapping


def test_critic_role_gets_verl_training_worker():
    """The critic runs verl's unified TrainingWorker, not a legacy fsdp/megatron worker."""
    from psrl.trainer.ppo.utils import PSRL_Role
    from verl.workers.engine_workers import TrainingWorker

    runner = _task_runner()
    config = OmegaConf.create({"critic": {"enable": True}, "algorithm": {"adv_estimator": "gae"}})
    runner.add_critic_worker(config)

    _assert_role_runs(runner.role_worker_mapping[PSRL_Role.Critic], TrainingWorker)
    assert runner.mapping[PSRL_Role.Critic] == ["train_pool"]


def test_critic_role_absent_when_critic_disabled():
    """GRPO-style runs need no critic, so the role must not claim a resource pool."""
    from psrl.trainer.ppo.utils import PSRL_Role

    runner = _task_runner()
    config = OmegaConf.create({"critic": {"enable": False}, "algorithm": {"adv_estimator": "grpo"}})
    runner.add_critic_worker(config)

    assert PSRL_Role.Critic not in runner.role_worker_mapping
