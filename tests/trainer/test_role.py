import enum

import pytest
from psrl.trainer.ppo.utils import PSRL_Role

pytestmark = pytest.mark.cpu_test


class TestPSRLRole:
    def test_roles_cover_every_disaggregated_component(self):
        """Pin the roles the trainer actually assigns resource pools to.

        A hardcoded superset rots whenever a component is added, so this asserts
        the roles that `ray_trainer` and `main_ppo` reference by name. A new role
        must be used somewhere before it is listed here.
        """
        required = {
            "Actor",
            "Rollout",
            "Critic",
            "RefPolicy",
            "RewardModel",
            "Validate",
            "TeacherModel",
            "DummyPolicy",
        }
        assert {r.name for r in PSRL_Role} == required

    def test_no_fused_actor_rollout_role(self):
        """PSRL disaggregates generation from training, so a fused role is meaningless."""
        names = {r.name for r in PSRL_Role}
        assert "ActorRollout" not in names
        assert "ActorRolloutRef" not in names

    def test_role_values_are_unique(self):
        values = [r.value for r in PSRL_Role]
        assert len(values) == len(set(values))

    def test_roles_are_enum_members(self):
        for role in PSRL_Role:
            assert PSRL_Role[role.name] == role

    def test_role_is_usable_as_dict_key(self):
        role_map = {role: role.name for role in PSRL_Role}
        assert role_map[PSRL_Role.Actor] == "Actor"
        assert role_map[PSRL_Role.Validate] == "Validate"

    def test_role_is_hashable_and_distinct_from_plain_ints(self):
        """Resource-pool maps key on the enum, so an int must not collide with a role."""
        assert PSRL_Role.Actor != PSRL_Role.Actor.value
        assert isinstance(PSRL_Role.Actor, enum.Enum)
