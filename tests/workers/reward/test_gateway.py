"""CPU tests for the reward-model gateway's router configuration.

`RewardModelGateway` is a Ray actor, so its routing policy is asserted through
`build_reward_router_args`, the plain function `_init_router_args` delegates to.
"""

import pytest
from omegaconf import OmegaConf
from psrl.workers.gen.smg_adapter import build_reward_router_args

pytestmark = pytest.mark.cpu_test


def _args(**overrides):
    config = OmegaConf.create({"psrl": {"logging_path": "/tmp/psrl_test_logs"}})
    kwargs = {"host": "127.0.0.1", "port": 8300, "prometheus_port": None}
    kwargs.update(overrides)
    return build_reward_router_args(config, **kwargs)


def test_router_uses_round_robin_without_psrl_routing():
    """A reward model has no model-version gating, so PSRL worker selection is wrong for it."""
    args = _args()
    assert args.policy == "round_robin"
    assert args.enable_routing_loop is False
    assert args.worker_selection_strategy == "naive"


def test_router_is_not_dp_aware():
    """Reward replicas are addressed as whole workers, not per dp rank."""
    assert _args().dp_aware is False


def test_router_disables_pd_disaggregation():
    """Reward scoring is a single forward pass, so there is no prefill/decode split."""
    args = _args()
    assert args.pd_disaggregation is False
    assert not args.prefill_urls
    assert not args.decode_urls


def test_router_binds_the_requested_host_and_port():
    args = _args(host="10.0.0.5", port=9999)
    assert args.host == "10.0.0.5"
    assert args.port == 9999


def test_prometheus_port_is_passed_through():
    assert _args(prometheus_port=4100).prometheus_port == 4100
    assert _args(prometheus_port=None).prometheus_port is None


def test_router_uses_grpc_connection_mode():
    """PSRL serves vLLM over gRPC, so the reward router must match."""
    assert _args().connection_mode == "grpc"
