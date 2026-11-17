"""Run common development, freeze T10 controls, then evaluate hidden/OOD splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from rebranch.baselines import BaselineResult
from rebranch.baselines.strong import run_parameter_only
from rebranch.baselines.t10_controls import run_t10_control_development
from rebranch.core.development import admission_cases
from rebranch.eval.metrics import EvalMetrics, compute_metrics
from rebranch.eval.runner import _save_run_artifacts, load_task_backend
from rebranch.exec.evaluator import Evaluator
from rebranch.sim.backend import TaskBackend


JOBS = (
    ("t10_hidden_mass_balance_beam_a", "t10_fixed_blind"),
    ("t10_hidden_mass_balance_beam_a", "t10_mass_balance"),
    ("t10_hidden_mass_balance_beam_b", "t10_tilt_feedback"),
    ("t10_hidden_mass_balance_beam_a", "t10_parameter_only"),
)


def _write_jsonl(path: Path, record: dict) -> None:
    """Append one complete artifact record to a JSONL file."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _evaluate_static_split(
    backend: TaskBackend,
    result: BaselineResult,
    cases: list[dict[str, Any]],
    *,
    seed: int,
    workers: int,
    trace_path: Path,
    extra_primitives: Sequence[str] = (),
) -> EvalMetrics:
    """Save the existing nominal check and split rollouts without rerunning episodes."""
    with Evaluator(backend, workers=1, extra_primitives=extra_primitives) as evaluator:
        nominal = evaluator.run(result.policy_source, backend.param_space.nominal(), seed=seed)
    trace_path.write_text("", encoding="utf-8")
    _write_jsonl(trace_path, {"kind": "nominal", "trace": asdict(nominal)})

    # Preserve runner._evaluate_split's noise route and per-episode API budget.
    seeded_cases = [(theta, seed * 1000 + idx) for idx, theta in enumerate(cases)]
    with Evaluator(backend, workers=workers, extra_primitives=extra_primitives) as evaluator:
        traces = evaluator.run_many(result.policy_source, seeded_cases)
    for idx, trace in enumerate(traces):
        _write_jsonl(trace_path, {"kind": "test", "case_index": idx, "trace": asdict(trace)})
    return compute_metrics(
        backend,
        traces,
        policy_sources=[result.policy_source] * len(traces),
        sim_predicted_success=[nominal.success] * len(traces),
        dev_llm_calls=result.dev_llm_calls,
        dev_tokens=result.dev_prompt_tokens + result.dev_completion_tokens,
        runtime_llm_calls_per_ep=[0] * len(traces),
        runtime_tokens_per_ep=[0] * len(traces),
        n_rollouts_dev=result.n_rollouts,
    )


def main() -> None:
    """Develop and freeze four controls, then save final metrics and complete traces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-rollouts", type=int, required=True,
                        help="Shared development cap for each parameter-only job")
    parser.add_argument("--n-hidden", type=int, required=True)
    parser.add_argument("--n-boundary", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/repair_validation/t10_baseline_pilot_v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    backend_a = load_task_backend("t10_hidden_mass_balance_beam_a")
    shared_dev_thetas = admission_cases(backend_a, seed=args.seed, n=24)
    dev_cases = [(theta, 10000 + i) for i, theta in enumerate(shared_dev_thetas)]
    results = []

    for task_name, method in JOBS:
        job_start = time.monotonic()
        print(f"JOB_START {task_name} {method}", flush=True)
        backend = load_task_backend(task_name)
        if method == "t10_parameter_only":
            result = run_parameter_only(
                backend,
                seed=args.seed,
                workers=args.workers,
                max_rollouts=args.max_rollouts,
                validation_n=24,
            )
        else:
            result = run_t10_control_development(
                backend,
                method,
                dev_cases,
                seed=args.seed,
                workers=args.workers,
                max_rollouts=24,
            )
        results.append((task_name, method, backend, result))
        source_hash = hashlib.sha256(result.policy_source.encode()).hexdigest()
        _write_jsonl(output_dir / "development.jsonl", {
            "task": task_name,
            "method": method,
            "seed": args.seed,
            "policy_sha256": source_hash,
            "n_rollouts": result.n_rollouts,
            "round_logs": result.round_logs,
            "extra": result.extra,
        })
        source_path = output_dir / "development_sources" / f"{task_name}__{method}.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(result.policy_source, encoding="utf-8")
        dev_sim_time_s = (result.extra["development_costs"]["sim_time_s"]
                          if method == "t10_parameter_only" else result.extra["dev_sim_time_s"])
        print(
            f"JOB_END {task_name} {method} rollouts={result.n_rollouts} "
            f"sim_time_s={dev_sim_time_s:.3f} "
            f"elapsed_s={time.monotonic() - job_start:.3f}",
            flush=True,
        )

    frozen_rows = []
    for task_name, method, _, result in results:
        source_hash = hashlib.sha256(result.policy_source.encode()).hexdigest()
        frozen_rows.append({
            "task": task_name,
            "method": method,
            "seed": args.seed,
            "policy_sha256": source_hash,
            "policy_source": result.policy_source,
        })
    (output_dir / "frozen_sources.json").write_text(
        json.dumps(frozen_rows, indent=2), encoding="utf-8"
    )

    hidden_cases = backend_a.hidden_test_set(seed=2026, n=args.n_hidden)
    boundary_cases = backend_a.ood_test_set(seed=3026, n=args.n_boundary)
    trace_dir = output_dir / "final_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for task_name, method, backend, result in results:
        trace_paths = {
            split: trace_dir / f"{task_name}__{method}__seed{args.seed}__{split}.jsonl"
            for split in ("hidden", "boundary")
        }
        split_metrics = {
            "hidden": _evaluate_static_split(
                backend, result, hidden_cases, seed=args.seed, workers=args.workers,
                trace_path=trace_paths["hidden"],
                extra_primitives=("get_mass",) if task_name.endswith("_a") else (),
            ),
            "boundary": _evaluate_static_split(
                backend, result, boundary_cases, seed=args.seed + 500, workers=args.workers,
                trace_path=trace_paths["boundary"],
                extra_primitives=("get_mass",) if task_name.endswith("_a") else (),
            ),
        }
        result.extra["final_trace_files"] = {
            split: str(path.relative_to(output_dir)) for split, path in trace_paths.items()
        }
        actual_test = {
            "hidden_cases": len(hidden_cases),
            "boundary_cases": len(boundary_cases),
            "nominal_checks": 2,
            "total_rollouts": len(hidden_cases) + len(boundary_cases) + 2,
        }
        result.extra["actual_test_rollouts_by_split"] = actual_test
        budget_info = {
            "protocol": "t10-v2-noslip2",
            "noslip_iterations": 2,
            "max_dev_rollouts": args.max_rollouts,
            "actual_dev_rollouts": result.n_rollouts,
            "max_llm_calls": 0,
            "dev_llm_calls": result.dev_llm_calls,
            "actual_test_rollouts": actual_test,
            "workers": args.workers,
            "development_seed": args.seed,
            "hidden_theta_seed": 2026,
            "boundary_theta_seed": 3026,
            "hidden_noise_seed_base": args.seed * 1000,
            "hidden_noise_seed_formula": "development_seed * 1000 + case_index",
            "boundary_noise_seed_base": (args.seed + 500) * 1000,
            "boundary_noise_seed_formula": "(development_seed + 500) * 1000 + case_index",
            "nominal_noise_seeds": {"hidden": args.seed, "boundary": args.seed + 500},
            "policy_sha256": hashlib.sha256(result.policy_source.encode()).hexdigest(),
        }
        _save_run_artifacts(
            output_dir, task_name, method, "full", args.seed, result,
            split_metrics, budget_info,
        )

    summary = {
        "protocol": "t10-v2-noslip2",
        "jobs": list(JOBS),
        "development_cases": 24,
        "max_parameter_rollouts_per_job": args.max_rollouts,
        "hidden_cases_per_job": args.n_hidden,
        "boundary_cases_per_job": args.n_boundary,
        "actual_final_rollouts_total": len(JOBS) * (args.n_hidden + args.n_boundary + 2),
        "llm_calls": 0,
        "source_frozen_before_final_test": True,
        "wall_time_s": time.monotonic() - start,
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
