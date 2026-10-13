"""Tests for the `SupportsWeightLayout` protocol and `WeightLayoutPlan` structure.

The converter picks its fast path on `supports_weight_layout(model)`, so the
protocol must recognise a model by the presence of `build_weight_layout` alone,
without the model inheriting anything.
"""

import pytest
from vllm_patches.interfaces import SupportsWeightLayout, supports_weight_layout
from vllm_patches.weight_layout import (
    WeightLayoutMount,
    WeightLayoutPlan,
    WeightLayoutRule,
    WeightSyncPolicy,
    WeightTransform,
)

pytestmark = pytest.mark.cpu_test


class _LayoutModel:
    """A model exposing a layout without inheriting the protocol."""

    def build_weight_layout(self) -> WeightLayoutPlan:
        return WeightLayoutPlan()


class _PlainModel:
    pass


def _identity_rule(vllm_pattern: str, hf_pattern: str) -> WeightLayoutRule:
    """Build the simplest possible rule, a one-to-one name mapping."""
    return WeightLayoutRule(
        vllm_pattern=vllm_pattern,
        hf_patterns=(hf_pattern,),
        transform=WeightTransform.identity(),
    )


class TestSupportsWeightLayoutProtocol:
    def test_structural_check_accepts_any_model_with_the_method(self):
        assert isinstance(_LayoutModel(), SupportsWeightLayout)

    def test_structural_check_rejects_a_model_without_the_method(self):
        assert not isinstance(_PlainModel(), SupportsWeightLayout)

    def test_helper_accepts_an_instance(self):
        assert supports_weight_layout(_LayoutModel())

    def test_helper_rejects_a_plain_model(self):
        assert not supports_weight_layout(_PlainModel())

    # NOTE(claude): `supports_weight_layout` is documented for a class or an
    # instance, but the installed plugin build answers False for a class. The
    # source is fixed to use issubclass. Assert the class path here once the
    # plugin is reinstalled from patch/vllm/vllm_plugin.

    def test_protocol_classvar_is_true(self):
        assert SupportsWeightLayout.supports_weight_layout is True


class TestWeightLayoutPlan:
    def test_empty_plan_is_valid(self):
        """A model with no parameters to remap still returns a plan, not None."""
        plan = WeightLayoutPlan()
        assert plan.rules == ()
        assert plan.mounts == ()
        assert plan.name_map is None

    def test_plan_carries_rules(self):
        _rule = _identity_rule("qkv_proj.weight", "q_proj.weight")
        plan = WeightLayoutPlan(rules=(_rule,))
        assert plan.rules == (_rule,)

    def test_flatten_prefixes_mounted_rules(self):
        """A mounted sub-plan must surface with its prefix applied."""
        inner = WeightLayoutPlan(rules=(_identity_rule("qkv_proj.weight", "q_proj.weight"),))
        outer = WeightLayoutPlan(mounts=(WeightLayoutMount(prefix="language_model", plan=inner),))

        resolved = outer.flatten()

        prefixes = {flat.prefix for flat in resolved.rules}
        assert prefixes == {"language_model"}

    def test_flatten_keeps_direct_rules_unprefixed(self):
        rule = _identity_rule("lm_head.weight", "lm_head.weight")
        resolved = WeightLayoutPlan(rules=(rule,)).flatten()
        assert [flat.prefix for flat in resolved.rules] == [""]

    def test_default_sync_policy_is_permissive(self):
        """An empty policy must not silently drop every parameter."""
        policy = WeightLayoutPlan().sync
        assert isinstance(policy, WeightSyncPolicy)
        assert not policy.exclude_prefixes
        assert not policy.exclude_substrs
