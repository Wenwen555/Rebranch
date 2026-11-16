"""Fixed compliant and parameter-only controls for the formal benchmark runner."""

from __future__ import annotations

from collections import Counter
import re
import time
from typing import Any, Sequence

import numpy as np

from rebranch.baselines import BaselineResult
from rebranch.core.development import admission_cases
from rebranch.core.lifting import ConstSpec, lift_constants
from rebranch.core.retune import Retuner
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace
from rebranch.sim.backend import TaskBackend
from rebranch.baselines.t10_controls import parameterized_blind_source


_SUPPORTED_PARAMETER_TASKS = {"t2_cup", "t4_cabinet", "t5_drawer"}
_T10_PARAMETER_TASKS = {
    "t10_hidden_mass_balance_beam_a",
    "t10_hidden_mass_balance_beam_b",
}
_COMPLIANT_AXES = {"t4_cabinet": "yzXYZ", "t5_drawer": "yzXYZ"}


class _T10CostEvaluator(Evaluator):
    """Accumulate costs for every T10 Retuner rollout, including rejected candidates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sim_time_s = 0.0
        self.api_calls = 0
        self.mass_calls = 0

    def run_many(self, source, cases, seed=0):
        traces = super().run_many(source, cases, seed=seed)
        self.sim_time_s += sum(trace.sim_time for trace in traces)
        self.api_calls += sum(len(trace.calls) for trace in traces)
        self.mass_calls += sum(
            call.name == "get_mass" for trace in traces for call in trace.calls
        )
        return traces


def _rank(traces: Sequence[Trace]) -> tuple[int, float]:
    if not traces:
        return 0, 0.0
    return sum(bool(t.success) for t in traces), sum(float(t.score) for t in traces) / len(traces)


def compare_validation(
    incumbent: Sequence[Trace], candidate: Sequence[Trace]
) -> tuple[bool, tuple[int, float], tuple[int, float], list[int]]:
    """Choose by fixed-set (success count, mean score); report per-case regressions."""
    if len(incumbent) != len(candidate):
        raise ValueError("incumbent and candidate must use the same validation cases")
    old_rank, new_rank = _rank(incumbent), _rank(candidate)
    regressions = [
        i for i, (old, new) in enumerate(zip(incumbent, candidate))
        if old.success and not new.success
    ]
    return new_rank > old_rank, old_rank, new_rank, regressions


def fixed_compliant_source(backend: TaskBackend) -> str:
    """Build a fixed pull policy whose declared 0.125 m/0.04 m/s are used."""
    axes = _COMPLIANT_AXES.get(backend.name)
    if axes is None:
        raise ValueError(f"fixed_compliant supports T4/T5, got {backend.name!r}")
    source = backend.draft_policy
    source, n_dist = re.subn(
        r"(?m)^(PULL_DIST\s*=\s*)[-+]?(?:\d+\.?\d*|\.\d+)",
        r"\g<1>0.125", source, count=1,
    )
    source, n_speed = re.subn(
        r"(?m)^(PULL_SPEED\s*=\s*)[-+]?(?:\d+\.?\d*|\.\d+)",
        r"\g<1>0.04", source, count=1,
    )
    source, n_force = re.subn(
        r"(?m)^(GRIP_FORCE\s*=\s*)[-+]?(?:\d+\.?\d*|\.\d+)",
        r"\g<1>18.0", source, count=1,
    )
    source, n_action = re.subn(
        r"(?m)^    api\.move_to\(\[h\[0\] - PULL_DIST, h\[1\], h\[2\]\], speed=PULL_SPEED\)$",
        f'    api.move_compliant([h[0] - PULL_DIST, h[1], h[2]], free_axes="{axes}", speed=PULL_SPEED)',
        source, count=1,
    )
    if (n_dist, n_speed, n_force, n_action) != (1, 1, 1, 1):
        raise ValueError(f"draft template mismatch for fixed compliant policy: {(n_dist, n_speed, n_force, n_action)}")
    return source


def run_fixed_compliant(backend: TaskBackend) -> BaselineResult:
    source = fixed_compliant_source(backend)
    params = {c.name: c.initial for c in lift_constants(source)}
    return BaselineResult(
        method="fixed_compliant",
        policy_source=source,
        n_rollouts=0,
        dev_llm_calls=0,
        dev_prompt_tokens=0,
        dev_completion_tokens=0,
        round_logs=[{
            "round": 0,
            "kind": "fixed_control",
            "development_rollouts": 0,
            "parameters": params,
            "action": "move_compliant",
            "stop_reason": "fixed_control_has_zero_development_budget",
        }],
        program_snapshots=[{"round": 0, "source": source}],
        extra={
            "baseline_type": "fixed_compliant",
            "dev_rollout_budget": 0,
            "parameters": params,
            "pull_dist_m": 0.125,
            "pull_speed_m_s": 0.04,
            "grip_force_n": 18,
            "compliant_free_axes": _COMPLIANT_AXES[backend.name],
            "selection_rule": "none_fixed_control",
        },
    )


def _failure_counts(traces: Sequence[Trace]) -> dict[str, int]:
    counts = Counter(
        (t.failure or "none") + " | " + (t.error or "none") for t in traces
    )
    return {k: int(v) for k, v in sorted(counts.items())}


def run_parameter_only(
    backend: TaskBackend,
    *,
    seed: int = 0,
    workers: int = 1,
    max_rollouts: int = 2000,
    validation_n: int = 24,
    training_cases_per_restart: int = 24,
    optimizer_restart_rollouts: int = 2000,
    compliant: bool = False,
) -> BaselineResult:
    """Tune constants only; rank every proposal on one fixed 24-case dev set.

    Training samples are independently drawn from the uniform active parameter
    space at each restart. Training and fixed-set validation consume the same
    evaluator cap. The validation rule intentionally does not preserve each
    incumbent success as required by ReBranch's stricter admission gate.
    """
    is_t10 = backend.name in _T10_PARAMETER_TASKS
    if backend.name not in _SUPPORTED_PARAMETER_TASKS | _T10_PARAMETER_TASKS:
        raise ValueError(f"parameter_only does not support {backend.name!r}")
    if is_t10 and compliant:
        raise ValueError("T10 has no compliant parameter-only variant")
    method = "t10_parameter_only" if is_t10 else (
        "compliant_parameter_only" if compliant else "parameter_only"
    )
    source = (
        parameterized_blind_source() if is_t10
        else fixed_compliant_source(backend) if compliant
        else backend.draft_policy
    )
    consts: list[ConstSpec] = lift_constants(source)
    if not consts:
        raise ValueError(f"{method} needs annotated constants in its fixed draft structure")
    if validation_n != 24:
        raise ValueError("the formal parameter-only validation set is fixed at 24 cases")
    if training_cases_per_restart < 1 or optimizer_restart_rollouts < 1:
        raise ValueError("training set and optimizer restart budget must be positive")

    start = time.monotonic()
    initial_source = source
    val_cases = admission_cases(backend, seed=seed, n=validation_n)
    train_rng = np.random.default_rng(seed + 1)
    train_sample_seed = seed + 1
    round_logs: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = [{"round": 0, "source": initial_source}]
    stop_reason = "development_rollout_cap"
    accepted_restarts = 0
    rejected_restarts = 0

    evaluator_type = _T10CostEvaluator if is_t10 else Evaluator
    with evaluator_type(backend, workers=workers, budget_limit=max_rollouts) as ev:
        if max_rollouts < validation_n:
            incumbent_traces: list[Trace] = []
            stop_reason = "cap_below_fixed_validation_cost"
        else:
            incumbent_traces = ev.run_many(source, val_cases, seed=10000)
            incumbent_rank = _rank(incumbent_traces)
            validation_incumbent_cost = ev.n_rollouts
            restart = 0
            while ev.n_rollouts + training_cases_per_restart + validation_n <= max_rollouts:
                train_cases = [
                    (backend.param_space.sample(train_rng), train_sample_seed + restart * 1000 + i)
                    for i in range(training_cases_per_restart)
                ]
                remaining_before = max_rollouts - ev.n_rollouts
                train_cap = min(optimizer_restart_rollouts, remaining_before - validation_n)
                before_train = ev.n_rollouts
                optimizer = Retuner(
                    ev,
                    max_iters=1000,
                    popsize=8,
                    seed=seed + 63000 + restart,
                    max_rollouts=train_cap,
                    budget_reserve=validation_n,
                )
                tuned = optimizer.optimize(source, consts, train_cases)
                train_used = ev.n_rollouts - before_train
                candidate = tuned.best_source
                candidate_validation_cost = 0
                accepted = False
                old_rank = _rank(incumbent_traces)
                new_rank = old_rank
                regressions: list[int] = []
                candidate_traces = incumbent_traces
                if candidate != source:
                    before_validation = ev.n_rollouts
                    candidate_traces = ev.run_many(candidate, val_cases, seed=10000)
                    candidate_validation_cost = ev.n_rollouts - before_validation
                    accepted, old_rank, new_rank, regressions = compare_validation(
                        incumbent_traces, candidate_traces
                    )
                    if accepted:
                        source = candidate
                        consts = lift_constants(source)
                        incumbent_traces = candidate_traces
                        accepted_restarts += 1
                        snapshots.append({"round": restart + 1, "source": source})
                    else:
                        rejected_restarts += 1

                round_logs.append({
                    "round": restart + 1,
                    "training_sample_seed": train_sample_seed + restart * 1000,
                    "training_cases": training_cases_per_restart,
                    "training_rollouts": train_used,
                    "optimizer": "shared Retuner",
                    "optimizer_restart_cap": train_cap,
                    "optimizer_evaluations": tuned.n_evals,
                    "optimizer_generations": optimizer.generations_completed,
                    "optimizer_stop_reason": (
                        "training_margin_reached" if tuned.success_all and tuned.worst_score >= 0.25
                        else "restart_rollout_cap" if train_used >= train_cap
                        else "no_further_optimizer_candidate"
                    ),
                    "training_rank": [tuned.n_success, tuned.mean_score] if tuned.traces else None,
                    "training_failure_counts": _failure_counts(tuned.traces),
                    "validation_rollouts": candidate_validation_cost,
                    "validation_noise_seed": 10000,
                    "old_validation_rank": list(old_rank),
                    "new_validation_rank": list(new_rank),
                    "validation_success_regression_indices": regressions,
                    "accepted_by_fixed_validation_rank": accepted,
                    "parameters": {c.name: c.initial for c in consts} if accepted else {
                        c.name: c.initial for c in lift_constants(candidate)
                    },
                    "stop_reason": "continued_within_shared_cap",
                })
                restart += 1
                if train_used == 0:
                    stop_reason = "no_optimizer_rollout_affordable"
                    break

            if ev.n_rollouts + training_cases_per_restart + validation_n > max_rollouts:
                stop_reason = "remaining_budget_cannot_fund_training_and_validation"

        dev_rollouts = ev.n_rollouts
        final_rank = _rank(incumbent_traces)
        validation_cost = validation_incumbent_cost if max_rollouts >= validation_n else 0
        t10_development_costs = ({
            "sim_time_s": ev.sim_time_s,
            "api_calls": ev.api_calls,
            "mass_calls": ev.mass_calls,
        } if is_t10 else None)

    final_params = {c.name: c.initial for c in lift_constants(source)}
    extra = {
        "baseline_type": method,
        "dev_rollout_budget": max_rollouts,
        "actual_dev_rollouts": dev_rollouts,
        "development_rollouts": {
            "fixed_validation_incumbent": validation_cost,
            "training_optimizer": dev_rollouts - validation_cost - sum(
                int(r["validation_rollouts"]) for r in round_logs
            ),
            "fixed_validation_candidates": sum(int(r["validation_rollouts"]) for r in round_logs),
            "total": dev_rollouts,
        },
        "validation": {
            "n_cases": validation_n,
            "evaluated_cases": len(incumbent_traces),
            "case_seed": seed,
            "noise_seed": 10000,
            "final_success_count": final_rank[0] if validation_cost else None,
            "final_mean_score": final_rank[1] if validation_cost else None,
            "selection_rule": "strict lexicographic improvement in (success_count, mean_score); per-case regressions are reported but do not veto",
        },
        "training": {
            "sampling": "uniform over active ParamSpace; fresh independent sample each restart",
            "sample_seed": train_sample_seed,
            "cases_per_restart": training_cases_per_restart,
            "optimizer_restart_rollout_cap": optimizer_restart_rollouts,
            "accepted_restarts": accepted_restarts,
            "rejected_restarts": rejected_restarts,
        },
        "initial_parameters": {
            c.name: c.initial for c in lift_constants(
                fixed_compliant_source(backend) if compliant else backend.draft_policy
            )
        },
        "final_parameters": final_params,
        "stop_reason": stop_reason,
        "dev_llm_calls": 0,
        "structure_policy": "fixed draft structure; only lifted annotated constants change",
        "comparison_difference": "unlike ReBranch admission, a success regression on one validation case does not veto a higher (success_count, mean_score) rank",
    }
    if is_t10:
        extra["control_identity"] = "custom_control"
        extra["development_costs"] = t10_development_costs
    return BaselineResult(
        method=method,
        policy_source=source,
        n_rollouts=dev_rollouts,
        dev_llm_calls=0,
        dev_prompt_tokens=0,
        dev_completion_tokens=0,
        round_logs=round_logs,
        program_snapshots=snapshots,
        wall_time_s=time.monotonic() - start,
        extra=extra,
    )
