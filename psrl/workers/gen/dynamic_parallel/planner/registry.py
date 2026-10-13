"""Cohort planner registry, following the reward-loop registry pattern."""

from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig

from psrl.workers.gen.dynamic_parallel.planner.base import CohortPlanner

__all__ = ["build_cohort_planner", "get_cohort_planner_cls", "register_cohort_planner"]

COHORT_PLANNER: dict[str, type[CohortPlanner]] = {}


def register_cohort_planner(name: str) -> Callable[[type[CohortPlanner]], type[CohortPlanner]]:
    """Decorator registering a `CohortPlanner` subclass under a config name.

    Args:
        name (str): The name used by `psrl.dynamic_parallel.planner`.

    Returns:
        Callable: A decorator registering the class under that name.
    """

    def decorator(cls: type[CohortPlanner]) -> type[CohortPlanner]:
        if name in COHORT_PLANNER and COHORT_PLANNER[name] is not cls:
            raise ValueError(f"Cohort planner {name} is already registered: {COHORT_PLANNER[name]} vs {cls}.")
        COHORT_PLANNER[name] = cls
        return cls

    return decorator


def get_cohort_planner_cls(name: str) -> type[CohortPlanner]:
    """Look up a registered `CohortPlanner` subclass.

    Args:
        name (str): The registered planner name.

    Returns:
        type[CohortPlanner]: The planner class.

    Raises:
        ValueError: If no planner is registered under that name.
    """
    _import_builtin_planners()
    if name not in COHORT_PLANNER:
        known = ", ".join(sorted(COHORT_PLANNER))
        raise ValueError(f"Unknown cohort planner {name!r}. Registered planners: {known}.")
    return COHORT_PLANNER[name]


def build_cohort_planner(config: DictConfig) -> CohortPlanner:
    """Build the planner named by `psrl.dynamic_parallel.planner`.

    Args:
        config (DictConfig): The `psrl.dynamic_parallel` node.

    Returns:
        CohortPlanner: A planner ready to be called at each boundary.
    """
    name = config.get("planner", "static")
    cls = get_cohort_planner_cls(name)
    return cls.from_config(config)


def _import_builtin_planners() -> None:
    """Import planner modules so their decorators run.

    Importing from `registry` would be circular, since each planner module
    imports `register_cohort_planner` from here.
    """
    from psrl.workers.gen.dynamic_parallel.planner import (
        scripted,  # noqa: F401
        static,  # noqa: F401
    )
