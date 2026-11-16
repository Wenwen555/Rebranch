"""Signature-grouped, binary-shrunk counterexample discovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rebranch.core.signature import FailureSignature, failure_signature
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import Trace
from rebranch.sim.params import ParamSpace


@dataclass
class Counterexample:
    """Representative failure instance shrunk toward nominal."""

    theta: dict[str, Any]
    trace: Trace
    signature: FailureSignature
    group_size: int
    severity: float
    distance_to_nominal: float
    seed: int = 0


def _failure_severity(trace: Trace) -> float:
    base = max(0.05, -float(trace.score))
    if trace.safety_violation:
        base += 0.5
    return float(base)


def find_counterexamples(
    evaluator: Evaluator,
    source: str,
    space: ParamSpace,
    nominal_trace: Trace,
    obs_scale: Mapping[str, float],
    rng: np.random.Generator,
    n_samples: int = 200,
    k_max: int = 6,
    n_bisect: int = 6,
    delta: float = 3.0,
) -> tuple[list[Counterexample], dict[str, Any]]:
    """Sample `space`, group failures by `FailureSignature`, binary-shrink, and rank.

    Returns `(top_counterexamples, stats)`.
    """
    if n_samples <= 0 or k_max <= 0:
        return [], {
            "n_samples": 0,
            "n_failures": 0,
            "n_groups": 0,
            "n_selected": 0,
            "failure_rate": 0.0,
            "n_bisect_rollouts": 0,
        }

    nominal_theta = space.nominal()
    cases: list[tuple[dict[str, Any], int]] = [
        (space.sample(rng), int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(n_samples))
    ]
    traces = evaluator.run_many(source, cases)

    groups: dict[FailureSignature, list[tuple[dict[str, Any], Trace, float, int]]] = (
        defaultdict(list)
    )
    n_failures = 0
    for (theta, seed), tr in zip(cases, traces):
        if not tr.success:
            n_failures += 1
            sig = failure_signature(tr, nominal_trace, obs_scale, delta=delta)
            dist = space.distance(nominal_theta, theta)
            groups[sig].append((theta, tr, dist, seed))

    if not groups:
        return [], {
            "n_samples": len(cases),
            "n_failures": 0,
            "n_groups": 0,
            "n_selected": 0,
            "failure_rate": 0.0,
            "n_bisect_rollouts": 0,
        }

    # Pick closest-to-nominal failure per signature group.
    state: list[dict[str, Any]] = []
    for sig, members in groups.items():
        theta_best, trace_best, dist_best, seed_best = min(
            members, key=lambda item: (item[2], item[1].score)
        )
        state.append(
            {
                "sig": sig,
                "group_size": len(members),
                "orig_theta": theta_best,
                "cur_theta": theta_best,
                "cur_trace": trace_best,
                "seed": seed_best,
                "lo": 0.0,
                "hi": 1.0,
            }
        )

    # Batch each bisection depth across all signature groups.
    n_bisect_rollouts = 0
    for _ in range(max(0, int(n_bisect))):
        trial_cases: list[tuple[dict[str, Any], int]] = []
        mids: list[float] = []
        for entry in state:
            mid = 0.5 * (entry["lo"] + entry["hi"])
            cand = space.interpolate(nominal_theta, entry["orig_theta"], mid)
            trial_cases.append((cand, entry["seed"]))
            mids.append(mid)

        trial_traces = evaluator.run_many(source, trial_cases)
        n_bisect_rollouts += len(trial_traces)

        for entry, (cand, _), mid, tr in zip(state, trial_cases, mids, trial_traces):
            if not tr.success:
                tr_sig = failure_signature(tr, nominal_trace, obs_scale, delta=delta)
                if tr_sig == entry["sig"]:
                    entry["hi"] = mid
                    entry["cur_theta"] = cand
                    entry["cur_trace"] = tr
                    continue
            entry["lo"] = mid

    ces: list[Counterexample] = []
    for entry in state:
        cur_theta = entry["cur_theta"]
        cur_trace = entry["cur_trace"]
        sev = _failure_severity(cur_trace)
        dist = space.distance(nominal_theta, cur_theta)
        ces.append(
            Counterexample(
                theta=cur_theta,
                trace=cur_trace,
                signature=entry["sig"],
                group_size=int(entry["group_size"]),
                severity=sev,
                distance_to_nominal=dist,
                seed=int(entry["seed"]),
            )
        )

    ces.sort(
        key=lambda ce: (-(ce.group_size * ce.severity), ce.distance_to_nominal)
    )
    selected = ces[: int(k_max)]
    stats = {
        "n_samples": len(cases),
        "n_failures": n_failures,
        "n_groups": len(groups),
        "n_selected": len(selected),
        "failure_rate": n_failures / len(cases),
        "n_bisect_rollouts": n_bisect_rollouts,
        "group_sizes": [ce.group_size for ce in selected],
    }
    return selected, stats


def find_ablation_counterexamples(
    evaluator: Evaluator,
    source: str,
    space: ParamSpace,
    nominal_trace: Trace,
    obs_scale: Mapping[str, float],
    rng: np.random.Generator,
    *,
    mode: str,
    n_samples: int = 200,
    k_max: int = 6,
    delta: float = 3.0,
) -> tuple[list[Counterexample], dict[str, Any]]:
    """Select counterexamples without signature grouping/bisection (`random_ce` or `worst_ce`)."""
    if mode not in ("random_ce", "worst_ce"):
        raise ValueError(f"Unsupported ablation CE mode: {mode!r}")

    nominal_theta = space.nominal()
    cases: list[tuple[dict[str, Any], int]] = [
        (space.sample(rng), int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(n_samples))
    ]
    traces = evaluator.run_many(source, cases)

    failures: list[tuple[dict[str, Any], Trace, int]] = [
        (theta, tr, seed) for (theta, seed), tr in zip(cases, traces) if not tr.success
    ]
    if not failures:
        return [], {
            "n_samples": len(cases),
            "n_failures": 0,
            "n_groups": 0,
            "n_selected": 0,
            "failure_rate": 0.0,
            "n_bisect_rollouts": 0,
        }

    if mode == "random_ce":
        chosen_idx = rng.choice(
            len(failures), size=min(int(k_max), len(failures)), replace=False
        )
        picked = [failures[int(i)] for i in chosen_idx]
    else:
        picked = sorted(failures, key=lambda item: (item[1].score, -_failure_severity(item[1])))[
            : int(k_max)
        ]

    selected: list[Counterexample] = []
    for theta, tr, seed in picked:
        sig = failure_signature(tr, nominal_trace, obs_scale, delta=delta)
        selected.append(
            Counterexample(
                theta=theta,
                trace=tr,
                signature=sig,
                group_size=1,
                severity=_failure_severity(tr),
                distance_to_nominal=space.distance(nominal_theta, theta),
                seed=seed,
            )
        )

    stats = {
        "n_samples": len(cases),
        "n_failures": len(failures),
        "n_groups": len(failures),
        "n_selected": len(selected),
        "failure_rate": len(failures) / len(cases),
        "n_bisect_rollouts": 0,
    }
    return selected, stats
