"""Baseline algorithms (§7) and shared SDK-only helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from rebranch.exec.sandbox import check_policy_source, run_policy
from rebranch.exec.trace import Trace
from rebranch.sim.backend import Episode, EpisodeResult, TaskBackend

_CONST_FLOAT_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*#\s*\[\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\]",
    re.MULTILINE,
)


class RuntimePolicyAgent(Protocol):
    """Protocol for baselines that adapt or execute policies at test runtime."""

    def evaluate_episode(
        self,
        backend: TaskBackend,
        theta: dict[str, Any],
        seed: int = 0,
    ) -> tuple[Trace, dict[str, Any]]:
        """Run one zero-shot test episode and return `(Trace, metadata)`."""


@dataclass
class BaselineResult:
    """Standardized result returned by every baseline method."""

    method: str
    policy_source: str
    n_rollouts: int
    dev_llm_calls: int
    dev_prompt_tokens: int
    dev_completion_tokens: int
    agent: RuntimePolicyAgent | None = None
    round_logs: list[dict[str, Any]] = field(default_factory=list)
    program_snapshots: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def extract_float_constants(source: str) -> list[tuple[str, float, float, float]]:
    """Extract top-level `NAME = val  # [lo, hi]` float constants."""
    out: list[tuple[str, float, float, float]] = []
    for m in _CONST_FLOAT_RE.finditer(source):
        name, val, lo, hi = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if lo <= hi:
            out.append((name, val, lo, hi))
    return out


def instantiate_float_constants(source: str, values: dict[str, float]) -> str:
    """Replace top-level `NAME = val  # [lo, hi]` constants in-place."""

    def _repl(m: re.Match[str]) -> str:
        name = m.group(1)
        lo = float(m.group(3))
        hi = float(m.group(4))
        if name not in values:
            return m.group(0)
        v = min(max(float(values[name]), lo), hi)
        if re.fullmatch(r"[-+]?\d+", m.group(2)):
            v = min(max(round(v), math.ceil(lo)), math.floor(hi))
            literal = str(int(v))
        else:
            literal = f"{v:.4f}"
        return f"{name} = {literal}  # [{m.group(3)}, {m.group(4)}]"

    return _CONST_FLOAT_RE.sub(_repl, source)


def tune_constants_cem(
    backend: TaskBackend,
    source: str,
    thetas: Sequence[dict[str, Any]],
    *,
    seed: int = 0,
    max_rollouts: int = 24,
    extra_primitives: Sequence[str] = (),
) -> tuple[str, int, float]:
    """Lightweight CEM constant optimizer for baseline parameter calibration."""
    consts = extract_float_constants(source)
    if not consts or not thetas or max_rollouts < len(thetas):
        return source, 0, -1.0

    rng = np.random.default_rng(seed)
    rollouts = 0

    def _eval_src(src: str) -> tuple[float, bool]:
        nonlocal rollouts
        if rollouts + len(thetas) > max_rollouts:
            return -1e9, False
        traces: list[Trace] = []
        for idx, th in enumerate(thetas):
            ep = backend.make_episode(th, seed=seed + idx)
            traces.append(
                run_policy(src, ep, th, seed=seed + idx, extra_primitives=extra_primitives)
            )
        rollouts += len(thetas)
        n_succ = sum(1 for t in traces if t.success)
        worst = min(t.score for t in traces)
        mean = sum(t.score for t in traces) / len(traces)
        return float(n_succ + 0.5 * worst + 0.1 * mean), n_succ == len(thetas)

    best_src = source
    best_obj, all_ok = _eval_src(source)
    if all_ok:
        return best_src, rollouts, best_obj

    lo = np.array([c[2] for c in consts], dtype=float)
    hi = np.array([c[3] for c in consts], dtype=float)
    mu = np.array([c[1] for c in consts], dtype=float)
    sigma = np.maximum((hi - lo) * 0.35, 1e-4)

    n_pop = max(4, min(8, (max_rollouts - rollouts) // max(len(thetas), 1)))
    if n_pop < 2:
        return best_src, rollouts, best_obj

    n_iters = max(1, (max_rollouts - rollouts) // max(n_pop * len(thetas), 1))
    for _ in range(n_iters):
        if rollouts + len(thetas) > max_rollouts:
            break
        pop: list[tuple[float, np.ndarray, str]] = []
        for _k in range(n_pop):
            if rollouts + len(thetas) > max_rollouts:
                break
            vec = np.clip(rng.normal(mu, sigma), lo, hi)
            vals = {c[0]: float(vec[i]) for i, c in enumerate(consts)}
            cand_src = instantiate_float_constants(source, vals)
            obj, ok = _eval_src(cand_src)
            pop.append((obj, vec, cand_src))
            if obj > best_obj:
                best_obj = obj
                best_src = cand_src
            if ok and obj > len(thetas) + 0.2:
                return best_src, rollouts, best_obj
        if len(pop) >= 2:
            pop.sort(key=lambda x: x[0], reverse=True)
            elites = np.stack([item[1] for item in pop[: max(2, len(pop) // 2)]], axis=0)
            mu = np.mean(elites, axis=0)
            sigma = np.maximum(np.std(elites, axis=0), (hi - lo) * 0.05)

    return best_src, rollouts, best_obj


def obs_distance_sq(
    obs_a: dict[str, float],
    obs_b: dict[str, float],
    obs_scale: dict[str, float],
) -> float:
    """Compute `obs_scale`-normalized squared Euclidean distance between snapshots."""
    total = 0.0
    for key, scale in obs_scale.items():
        if key in obs_a and key in obs_b:
            s = max(float(scale), 1e-6)
            d = (float(obs_a[key]) - float(obs_b[key])) / s
            total += d * d
    return total


def build_safe_probe_source(task: TaskBackend | str) -> str:
    """Explicit task-configured non-destructive probe, never a truncated action policy.

    A legacy string has no declared contact protocol and only observes free space.
    Task config uses object/approach/grasp, never hidden physical parameters.
    """
    config = getattr(task, "probe_config", None) if not isinstance(task, str) else None
    lines = ["# task-explicit-probe-v1", "def run(api):", "    probe_start = api.get_ee_pose()", "    api.zero_ee_wrench()"]
    if config:
        name, approach, grasp = config
        lines += [f"    p = api.get_object_pose({name!r})['pos']", "    api.open_gripper(0.068)"]
        if approach == "front":
            lines += ["    api.move_to([p[0]-0.06,p[1],p[2]], speed=0.04)"]
        else:
            lines += ["    api.move_to([p[0],p[1],p[2]+0.10], speed=0.04)"]
        if grasp:
            lines += ["    api.move_to(p, speed=0.03)", "    api.close_gripper(force=2.5)"]
    lines += ["    api.get_ee_wrench()", "    # PROBE_FEATURES_V2", "    api.get_gripper_state()"]
    if config:
        lines += ["    api.open_gripper(0.068)"]
        if approach == "front":
            lines += ["    api.move_to([p[0]-0.06,p[1],p[2]], speed=0.04)"]
        else:
            lines += ["    api.move_to([p[0],p[1],p[2]+0.10], speed=0.04)"]
    lines += ["    api.move_to(probe_start['pos'], quat=probe_start['quat'], speed=0.04)"]
    source = "\n".join(lines) + "\n"
    check_policy_source(source)
    return source


def probe_features(trace: Trace, source: str) -> dict[str, float]:
    """Snapshot at the explicit pre-reset measurement stage (SDK only).

    Bank construction and runtime must use this same source-defined stage.
    A failed probe has no substitute feature from the reset or failure state.
    """
    marker = source.splitlines().index("    # PROBE_FEATURES_V2") + 2
    for call in trace.calls:
        if call.lineno == marker and call.name == "get_gripper_state":
            return dict(call.obs)
    raise ValueError("Probe did not reach the pre-reset measurement stage")


def combine_two_phase_episode(
    ep: Episode,
    probe_source: str,
    action_source: str,
    theta: dict[str, Any],
    seed: int,
    *,
    extra_primitives: Sequence[str] = (),
    max_calls: int = 200,
) -> Trace:
    """Run a shared-budget probe/action episode and evaluate it exactly once."""
    t_probe = run_policy(
        probe_source, ep, theta, seed=seed, max_calls=max_calls,
        extra_primitives=extra_primitives, evaluate=False,
    )
    if t_probe.error is not None:
        return finalize_episode_trace(ep, t_probe)

    remaining = max_calls - len(t_probe.calls)
    if remaining <= 0:
        return finalize_episode_trace(
            ep, t_probe, execution_error=f"max_calls ({max_calls}) exhausted before action phase"
        )

    t_act = run_policy(
        action_source, ep, theta, seed=seed + 10000, max_calls=remaining,
        extra_primitives=extra_primitives, evaluate=False,
    )
    return merge_two_phase_traces(ep, t_probe, t_act, max_calls=max_calls)


def finalize_episode_trace(
    ep: Episode,
    trace: Trace,
    *,
    execution_error: str | None = None,
) -> Trace:
    """Run the one terminal evaluation and apply it without hiding execution errors."""
    eval_res = ep.evaluate()
    error = execution_error or trace.error
    if error is not None:
        trace.success = False
        trace.score = min(float(eval_res.score), -0.5)
        trace.failure = error
        trace.error = error
        trace.safety_violation = eval_res.safety_violation or trace.safety_violation
    else:
        trace.success = eval_res.success
        trace.score = float(eval_res.score)
        trace.failure = eval_res.failure
        trace.safety_violation = eval_res.safety_violation or trace.safety_violation
    trace.sim_time = float(getattr(ep, "sim_time", trace.sim_time))
    return trace


def merge_two_phase_traces(
    ep: Episode,
    t_probe: Trace,
    t_action: Trace,
    *,
    max_calls: int = 200,
) -> Trace:
    """Merge non-terminal phase traces and perform one episode-terminal evaluation."""
    merged_calls = list(t_probe.calls)
    offset = len(merged_calls)
    for c in t_action.calls:
        c.index += offset
        merged_calls.append(c)

    error = t_action.error
    if error is not None and error.startswith("max_calls ("):
        error = f"max_calls ({max_calls}) exceeded across episode phases"
    if len(merged_calls) > max_calls:
        error = f"max_calls ({max_calls}) exceeded across episode phases"
    trace = Trace(
        theta=dict(t_probe.theta),
        seed=int(t_probe.seed),
        calls=merged_calls,
        lines=list(t_probe.lines) + list(t_action.lines),
        success=False,
        score=-1.0,
        failure=error or t_action.failure,
        failure_call=(
            None if t_action.failure_call is None else t_action.failure_call + offset
        ),
        safety_violation=t_action.safety_violation or t_probe.safety_violation,
        error=error,
        sim_time=float(getattr(ep, "sim_time", 0.0)),
        wall_time=t_probe.wall_time + t_action.wall_time,
        protocol_version=t_probe.protocol_version,
    )
    return finalize_episode_trace(ep, trace)


__all__ = [
    "BaselineResult",
    "RuntimePolicyAgent",
    "build_safe_probe_source",
    "combine_two_phase_episode",
    "finalize_episode_trace",
    "merge_two_phase_traces",
    "extract_float_constants",
    "instantiate_float_constants",
    "obs_distance_sq",
    "tune_constants_cem",
]
