"""Run common development, freeze T10 controls, then evaluate hidden/OOD splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from rebranch.baselines.strong import run_parameter_only
from rebranch.baselines.t10_controls import run_t10_control_development
from rebranch.core.development import admission_cases
from rebranch.eval.runner import _evaluate_split, _save_run_artifacts, load_task_backend


JOBS = (
    ("t10_hidden_mass_balance_beam_a", "t10_fixed_blind"),
    ("t10_hidden_mass_balance_beam_a", "t10_parameter_only"),
    ("t10_hidden_mass_balance_beam_a", "t10_mass_balance"),
    ("t10_hidden_mass_balance_beam_b", "t10_tilt_feedback"),
)


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def main() -> None:
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
        print(
            f"JOB_END {task_name} {method} rollouts={result.n_rollouts} "
            f"sim_time_s={result.extra.get('dev_sim_time_s', 0.0):.3f} "
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
    for task_name, method, backend, result in results:
        split_metrics = {
            "hidden": _evaluate_split(
                backend, result, hidden_cases, seed=args.seed, workers=args.workers,
                extra_primitives=("get_mass",) if task_name.endswith("_a") else (),
            ),
            "boundary": _evaluate_split(
                backend, result, boundary_cases, seed=args.seed + 500, workers=args.workers,
                extra_primitives=("get_mass",) if task_name.endswith("_a") else (),
            ),
        }
        actual_test = {
            "hidden_cases": len(hidden_cases),
            "boundary_cases": len(boundary_cases),
            "nominal_checks": 2,
            "total_rollouts": len(hidden_cases) + len(boundary_cases) + 2,
        }
        result.extra["actual_test_rollouts_by_split"] = actual_test
        budget_info = {
            "max_dev_rollouts": args.max_rollouts,
            "actual_dev_rollouts": result.n_rollouts,
            "max_llm_calls": 0,
            "dev_llm_calls": result.dev_llm_calls,
            "actual_test_rollouts": actual_test,
            "workers": args.workers,
            "development_seed": args.seed,
            "hidden_theta_seed": 2026,
            "boundary_theta_seed": 3026,
            "hidden_noise_seed": args.seed,
            "boundary_noise_seed": args.seed + 500,
            "policy_sha256": hashlib.sha256(result.policy_source.encode()).hexdigest(),
        }
        _save_run_artifacts(
            output_dir, task_name, method, "full", args.seed, result,
            split_metrics, budget_info,
        )

    summary = {
        "protocol": "t10-baseline-pilot-v1",
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
