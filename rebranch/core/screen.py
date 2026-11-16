"""Sensitivity screening to filter irrelevant distractor parameters."""

from __future__ import annotations

from typing import Any

import numpy as np

from rebranch.exec.evaluator import Evaluator
from rebranch.sim.params import Continuous, Discrete, ParamSpace


def sensitivity_screen(
    evaluator: Evaluator,
    source: str,
    space: ParamSpace,
    rng: np.random.Generator,
    n_base: int = 8,
) -> list[str]:
    """Identify active parameters whose extreme perturbations flip policy success.

    Samples `n_base` base points from `space`, evaluates each active parameter at
    its extremes (or all choices for `Discrete`) while holding the rollout seed
    and all other parameters fixed, and retains parameters that cause at least
    one success/failure flip. If no parameter flips any base point, all active
    parameters in `space` are retained.
    """
    active_params = [p for p in space.params if p.name in space.active]
    if not active_params:
        return []
    if n_base <= 0:
        return [p.name for p in active_params]

    base_cases: list[tuple[dict[str, Any], int]] = []
    for _ in range(int(n_base)):
        theta = space.sample(rng)
        seed = int(rng.integers(0, 2**31 - 1))
        base_cases.append((theta, seed))

    base_traces = evaluator.run_many(source, base_cases)

    pert_cases: list[tuple[dict[str, Any], int]] = []
    pert_meta: list[tuple[int, str]] = []
    for base_idx, (base_theta, base_seed) in enumerate(base_cases):
        for param in active_params:
            if isinstance(param, Continuous):
                test_vals: tuple[Any, ...] = (param.lo, param.hi)
            elif isinstance(param, Discrete):
                test_vals = tuple(
                    c for c in param.choices if c != base_theta[param.name]
                )
            else:
                continue

            for val in test_vals:
                perturbed = dict(base_theta)
                perturbed[param.name] = val
                pert_cases.append((perturbed, base_seed))
                pert_meta.append((base_idx, param.name))

    sensitive: set[str] = set()
    if pert_cases:
        pert_traces = evaluator.run_many(source, pert_cases)
        for (base_idx, param_name), tr in zip(pert_meta, pert_traces):
            if tr.success != base_traces[base_idx].success:
                sensitive.add(param_name)

    if not sensitive:
        return [p.name for p in active_params]
    return [p.name for p in active_params if p.name in sensitive]
