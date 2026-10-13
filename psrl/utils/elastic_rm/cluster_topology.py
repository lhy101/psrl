from __future__ import annotations

import logging
import os

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.gpu_topology import ClusterTopology as BaseClusterTopology
from psrl.utils.gpu_topology import GPUSlot, InstanceIdentifier, InstanceInfo, InstanceStatus
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

__all__ = [
    "ClusterTopology",
    "GPUSlot",
    "InstanceIdentifier",
    "InstanceInfo",
    "InstanceStatus",
]


class ClusterTopology(BaseClusterTopology):
    """Cluster topology with elastic-RM's cross-role selection policy.

    The generic GPU bookkeeping lives in `psrl.utils.gpu_topology`. This subclass
    adds the two queries that are specific to trading GPUs between roles, such as
    rollout versus reward model.
    """

    # ── Conflict detection ───────────────────────────────────────────────

    def has_other_role_awaken_on_shared_gpu(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_id: RolloutInstanceId,
    ) -> bool:
        """Return True if any *awake* instance of a *different role* shares a GPU with this instance."""
        key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
        info = self.instances.get(key)
        if info is None or not info.gpu_slots:
            return False
        for gpu_slot in info.gpu_slots:
            for other_key in self.gpu_to_instances.get(gpu_slot, set()):
                if other_key.role == role:
                    continue
                other_info = self.instances.get(other_key)
                if other_info is not None and other_info.status == InstanceStatus.AWAKEN:
                    return True
        return False

    # ── Batch queries ────────────────────────────────────────────────────

    def select_non_conflicting_awake_ids(
        self,
        role: PSRL_Role,
        model_name: str,
        instance_ids: list[RolloutInstanceId],
        target_awake_num: int,
        min_awake_num: int = 1,
    ) -> list[RolloutInstanceId]:
        """Select up to *target_awake_num* instances whose GPUs don't overlap with already-AWAKEN instances.

        The occupied GPU set is derived from all instances currently marked AWAKEN
        in the topology, so no external state tracking is needed. Successive calls
        are safe as long as the caller marks the returned instances as AWAKEN (via
        ``set_status``) before the next call.

        Returns:
            list of selected RolloutInstanceIds.

        Raises:
            RuntimeError: if fewer than *min_awake_num* non-conflicting instances can be found.
        """
        occupied_gpu_slots = self.get_awaken_gpu_slots()
        selected_ids: list[RolloutInstanceId] = []
        for instance_id in sorted(instance_ids):
            key = InstanceIdentifier(role=role, model_name=model_name, instance_id=instance_id)
            info = self.instances.get(key)
            if info is None:
                continue
            gpu_slots = info.gpu_slots
            if gpu_slots and not gpu_slots.isdisjoint(occupied_gpu_slots):
                continue
            selected_ids.append(instance_id)
            occupied_gpu_slots.update(gpu_slots)
            if len(selected_ids) >= target_awake_num:
                break
        if len(selected_ids) < min_awake_num:
            raise RuntimeError(
                "Cannot select non-conflicting awake instances "
                f"(target={target_awake_num}, selected={len(selected_ids)}, min_required={min_awake_num})."
            )
        return selected_ids
