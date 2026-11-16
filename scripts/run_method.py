"""CLI to run a single method (ReBranch, ablation, or baseline) on one task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from rebranch.eval.runner import build_llm_client, run_single_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ReBranch, baseline, or ablation method.")
    parser.add_argument("--task", required=True, help="Task name (e.g. t2_cup, toy_branch)")
    parser.add_argument("--method", default="rebranch", help="Method name")
    parser.add_argument("--variant", default="full", help="ReBranch ablation variant")
    parser.add_argument("--config", default=None, help="Optional YAML override config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-hidden", type=int, default=50)
    parser.add_argument("--n-ood", type=int, default=10)
    parser.add_argument("--max-rollouts", type=int, default=2000)
    parser.add_argument("--validation-cases", type=int, default=24, help="Fixed parameter-only development validation set")
    parser.add_argument("--training-cases-per-restart", type=int, default=24, help="Independent uniform parameter-only training cases per Retuner restart")
    parser.add_argument("--optimizer-restart-rollouts", type=int, default=2000, help="Local Retuner cap per restart; all restarts share --max-rollouts")
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--runtime-llm-budget", type=int, default=100, help="Online proxy request attempts over all test episodes, separate from dev cap")
    parser.add_argument("--runtime-sim-budget-per-episode", type=int, default=24)
    parser.add_argument("--twin-mpc-prediction-horizon", type=int, default=3, help="Finite Twin-MPC forecast horizon in actions (must be >=2)")
    parser.add_argument("--physcap-max-turns", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-ce-samples", type=int, default=60)
    parser.add_argument("--provider", default="codex", choices=("codex", "openai", "mock"))
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--cache-dir", default=".cache/rebranch_llm")
    parser.add_argument("--output-dir", default="outputs/single")
    args = parser.parse_args()

    variant = args.variant
    method = args.method
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        method = cfg.get("method", method)
        overrides = cfg.get("rebranch_overrides") or {}
        for k, v in overrides.items():
            if v:
                variant = k

    llm = build_llm_client(
        provider=args.provider,
        model=args.model,
        effort=args.effort,
        cache_dir=args.cache_dir,
    )
    out = run_single_method(
        task_name=args.task,
        method=method,
        llm=llm,
        variant=variant,
        seed=args.seed,
        workers=args.workers,
        n_hidden=args.n_hidden,
        n_ood=args.n_ood,
        max_rollouts=args.max_rollouts,
        validation_cases=args.validation_cases,
        training_cases_per_restart=args.training_cases_per_restart,
        optimizer_restart_rollouts=args.optimizer_restart_rollouts,
        max_llm_calls=args.max_llm_calls,
        runtime_llm_budget=args.runtime_llm_budget,
        runtime_sim_budget_per_episode=args.runtime_sim_budget_per_episode,
        twin_mpc_prediction_horizon=args.twin_mpc_prediction_horizon,
        physcap_max_turns=args.physcap_max_turns,
        max_rounds=args.max_rounds,
        n_ce_samples=args.n_ce_samples,
        output_dir=args.output_dir,
    )
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
