"""Section 9 evaluation metrics and SDK-only trace summary helpers."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from rebranch.exec.trace import Trace
from rebranch.sim.backend import TaskBackend

SENSING_PROBE_CALLS: frozenset[str] = frozenset(
    {"get_ee_wrench", "get_gripper_state", "get_mass", "get_stiffness"}
)
SENSORY_SOURCE_CALLS: frozenset[str] = frozenset(
    {
        "get_ee_wrench",
        "get_gripper_state",
        "get_ee_pose",
        "get_object_pose",
        "move_until_contact",
        "get_mass",
        "get_stiffness",
    }
)


@dataclass
class EvalMetrics:
    """Complete §9 evaluation metrics for a method on a task split."""

    zero_shot_success_rate: float
    sim_false_positive_rate: float | None
    safety_violation_rate: float
    redundant_probe_rate: float | None
    missed_probe_rate: float | None
    dev_llm_calls: int
    dev_tokens: int
    runtime_llm_calls: float
    runtime_tokens: float | None
    mean_exec_time_s: float
    branch_discrimination_accuracy: float
    n_episodes: int
    mean_score: float
    n_rollouts_dev: int = 0
    prediction_coverage: float = 0.0
    n_rollouts_runtime_sim: int = 0
    runtime_sim_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return asdict(self)


def policy_has_sensory_branch(source: str) -> bool:
    """Return True if `source` contains an `if` branch conditioned on sensory API output."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    sensory_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            has_sense = any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in SENSORY_SOURCE_CALLS
                for sub in ast.walk(node.value)
            )
            if has_sense:
                for target in node.targets:
                    for name_node in ast.walk(target):
                        if isinstance(name_node, ast.Name):
                            sensory_vars.add(name_node.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Name) and sub.id in sensory_vars:
                    return True
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in SENSORY_SOURCE_CALLS
                ):
                    return True
    return False


def trace_used_probe(trace: Trace, policy_source: str | None = None) -> bool:
    """Return True if `trace` executed a sensory probe call or sensory branch."""
    for call in trace.calls:
        if call.name in SENSING_PROBE_CALLS:
            return True
    if policy_source is not None and policy_has_sensory_branch(policy_source):
        for call in trace.calls:
            if call.name in SENSORY_SOURCE_CALLS:
                return True
    return False


def compute_branch_discrimination_accuracy(
    backend: TaskBackend,
    traces: Sequence[Trace],
    policy_sources: Sequence[str] | None = None,
) -> float:
    """Compute path-cluster purity against offline `backend.latent_mode(theta)` labels.

    This diagnostic measures path separation, not correctness of a physical
    parameter estimate. Continuous feedback can succeed without separating paths.

    Traces that terminated prematurely with `trace.error is not None` do not earn
    branch discrimination credit on multi-mode tasks, preventing crashes from being
    mistaken for intentional branching.
    """
    if not traces:
        return 0.0

    modes = [str(backend.latent_mode(t.theta)) for t in traces]
    distinct_modes = set(modes)
    if len(distinct_modes) <= 1:
        return 1.0

    same_source = policy_sources is None or len(set(policy_sources)) <= 1
    clusters: dict[Any, list[str]] = defaultdict(list)

    for t, mode in zip(traces, modes):
        if t.error is not None:
            continue
        key: Any = t.path if same_source else tuple(c.name for c in t.calls)
        clusters[key].append(mode)

    correct = 0
    for cluster_modes in clusters.values():
        counts = Counter(cluster_modes)
        correct += max(counts.values())

    return correct / len(traces)


def compute_metrics(
    backend: TaskBackend,
    traces: Sequence[Trace],
    *,
    policy_sources: Sequence[str] | None = None,
    sim_predicted_success: Sequence[bool | None] | None = None,
    dev_llm_calls: int = 0,
    dev_tokens: int = 0,
    runtime_llm_calls_per_ep: Sequence[int] | None = None,
    runtime_tokens_per_ep: Sequence[int | None] | None = None,
    exec_times_s: Sequence[float] | None = None,
    n_rollouts_dev: int = 0,
    runtime_sim_rollouts_per_ep: Sequence[int] | None = None,
    runtime_sim_times_per_ep: Sequence[float] | None = None,
) -> EvalMetrics:
    """Compute all §9 metrics for a list of test `Trace`s on `backend`."""
    n = len(traces)
    if n == 0:
        return EvalMetrics(
            zero_shot_success_rate=0.0,
            sim_false_positive_rate=None,
            safety_violation_rate=0.0,
            redundant_probe_rate=0.0,
            missed_probe_rate=0.0,
            dev_llm_calls=int(dev_llm_calls),
            dev_tokens=int(dev_tokens),
            runtime_llm_calls=0.0,
            runtime_tokens=0.0,
            mean_exec_time_s=0.0,
            branch_discrimination_accuracy=0.0,
            n_episodes=0,
            mean_score=0.0,
            n_rollouts_dev=int(n_rollouts_dev),
        )

    zero_shot_sr = sum(1 for t in traces if t.success) / n
    safety_rate = sum(1 for t in traces if t.safety_violation is not None) / n
    mean_score = sum(float(t.score) for t in traces) / n

    preds = list(sim_predicted_success) if sim_predicted_success is not None else [None] * n
    known = [(p,t) for p,t in zip(preds,traces) if p is not None]
    sim_fp_rate = sum(p is True and not t.success for p,t in known)/len(known) if known else None
    prediction_coverage = len(known)/n

    probed_flags: list[bool] = []
    for idx, t in enumerate(traces):
        src = (
            policy_sources[idx]
            if policy_sources is not None and idx < len(policy_sources)
            else None
        )
        probed_flags.append(trace_used_probe(t, policy_source=src))

    # Tasks whose feedback is not recognized by this API/AST heuristic opt out.
    # Missing classification is unavailable, not evidence that every probe was missed.
    if not getattr(backend, "probe_metrics_available", True):
        redundant_probe_rate = None
        missed_probe_rate = None
    elif backend.requires_branching:
        redundant_probe_rate = 0.0
        missed_probe_rate = sum(1 for p in probed_flags if not p) / n
    else:
        redundant_probe_rate = sum(1 for p in probed_flags if p) / n
        missed_probe_rate = 0.0

    rt_calls = (
        sum(runtime_llm_calls_per_ep) / n
        if runtime_llm_calls_per_ep is not None and len(runtime_llm_calls_per_ep) == n
        else 0.0
    )
    rt_tokens = (
        (sum(runtime_tokens_per_ep) / n if all(x is not None for x in runtime_tokens_per_ep) else None)
        if runtime_tokens_per_ep is not None and len(runtime_tokens_per_ep) == n
        else 0.0
    )
    if exec_times_s is not None and len(exec_times_s) == n:
        mean_exec = sum(float(x) for x in exec_times_s) / n
    else:
        mean_exec = sum(float(t.sim_time) + float(t.wall_time) for t in traces) / n

    branch_acc = compute_branch_discrimination_accuracy(
        backend, traces, policy_sources=policy_sources
    )

    return EvalMetrics(
        zero_shot_success_rate=float(zero_shot_sr),
        sim_false_positive_rate=sim_fp_rate,
        safety_violation_rate=float(safety_rate),
        redundant_probe_rate=redundant_probe_rate,
        missed_probe_rate=missed_probe_rate,
        dev_llm_calls=int(dev_llm_calls),
        dev_tokens=int(dev_tokens),
        runtime_llm_calls=float(rt_calls),
        runtime_tokens=rt_tokens,
        mean_exec_time_s=float(mean_exec),
        branch_discrimination_accuracy=float(branch_acc),
        n_episodes=n,
        mean_score=float(mean_score),
        n_rollouts_dev=int(n_rollouts_dev),
        prediction_coverage=prediction_coverage,
        n_rollouts_runtime_sim=sum(runtime_sim_rollouts_per_ep or []),
        runtime_sim_time_s=sum(runtime_sim_times_per_ep or []),
    )


def _compact_sdk_value(value: Any) -> str:
    """Compact SDK values without losing tuples, arguments, or return fields."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, str)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        values = ", ".join(_compact_sdk_value(v) for v in value[:6])
        if len(value) > 6:
            values += ", …"
        left, right = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        return f"{left}{values}{right}"
    if isinstance(value, dict):
        body = ", ".join(f"{_compact_sdk_value(k)}: {_compact_sdk_value(v)}" for k, v in value.items())
        return "{" + body + "}"
    return repr(value)


def format_sdk_calls(trace: Trace, max_calls: int = 8) -> str:
    """Format recent SDK calls using their actual arguments, results and observations."""
    limit = max(0, int(max_calls))
    tail = trace.calls[-limit:] if limit else []
    lines = [f"Recent SDK calls (showing {len(tail)} of {len(trace.calls)}):"]
    for c in tail:
        obs = {k: v for k, v in c.obs.items() if k != "theta"}
        lines.append(
            f"  #{c.index} L{c.lineno} {c.name} args={_compact_sdk_value(c.args)} "
            f"kwargs={_compact_sdk_value(c.kwargs)} result={_compact_sdk_value(c.result)} "
            f"obs={_compact_sdk_value(obs)}"
        )
    return "\n".join(lines)


def summarize_trace_for_llm(trace: Trace, max_calls: int = 8) -> str:
    """Format a Trace into a concise SDK-only summary (never includes theta)."""
    status = "SUCCESS" if trace.success else "FAILED"
    lines = [
        f"Outcome: {status}, score={trace.score:.3f}, sim_time={trace.sim_time:.2f}s",
        f"Task failure: {trace.failure or 'none'}",
        f"Execution error: {trace.error or 'none'}",
    ]
    if trace.safety_violation:
        lines.append(f"Safety violation: {trace.safety_violation}")
    if trace.calls:
        lines.append(format_sdk_calls(trace, max_calls=max_calls))
    else:
        lines.append("No primitive calls were recorded.")
    return "\n".join(lines)
