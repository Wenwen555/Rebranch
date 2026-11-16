"""CLI to run a benchmark suite (M2 or M4) from YAML config and generate tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from rebranch.eval.runner import build_llm_client, run_benchmark_suite
from scripts.make_tables import generate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ReBranch benchmark suite.")
    parser.add_argument("--config", required=True, help="Path to YAML suite config")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated subset of tasks")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated subset of seeds")
    parser.add_argument("--output-dir", default=None, help="Optional override for output_dir")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument("--n-ood", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--tables-dir", default="outputs/tables")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    llm_cfg = cfg.get("llm") or {}
    eval_cfg = cfg.get("eval") or {}

    llm = build_llm_client(
        provider=str(llm_cfg.get("provider", "codex")),
        model=str(llm_cfg.get("model", "gpt-6.1-sol")),
        effort=str(llm_cfg.get("effort", "low")),
        cache_dir=str(llm_cfg.get("cache_dir", ".cache/rebranch_llm")),
    )

    tasks = list(cfg.get("tasks") or [])
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t in wanted]

    out_dir = Path(str(args.output_dir or cfg.get("output_dir", "outputs/suite")))
    run_benchmark_suite(
        tasks=tasks,
        methods=list(cfg.get("methods") or []),
        llm=llm,
        ablations=list(cfg.get("ablations") or []),
        ablation_tasks=(
            list(cfg["ablation_tasks"]) if cfg.get("ablation_tasks") is not None else None
        ),
        seeds=(
            [int(s) for s in args.seeds.split(",") if s.strip()]
            if args.seeds
            else [int(s) for s in (eval_cfg.get("seeds") or [0, 1, 2])]
        ),
        workers=int(args.workers or eval_cfg.get("workers", 16)),
        n_hidden=int(args.n_hidden if args.n_hidden is not None else eval_cfg.get("n_hidden", 50)),
        n_ood=int(args.n_ood if args.n_ood is not None else eval_cfg.get("n_ood", 10)),
        max_rollouts=int(eval_cfg.get("max_rollouts", 2000)),
        max_llm_calls=int(eval_cfg.get("max_llm_calls", 8)),
        runtime_llm_budget=int(eval_cfg.get("runtime_llm_budget", 100)),
        runtime_sim_budget_per_episode=int(eval_cfg.get("runtime_sim_budget_per_episode", 24)),
        max_rounds=int(eval_cfg.get("max_rounds", 3)),
        n_ce_samples=int(eval_cfg.get("n_ce_samples", 80)),
        output_dir=out_dir,
        resume=not args.no_resume,
    )

    if args.tables_dir and not args.no_tables:
        generate_tables([out_dir], Path(args.tables_dir))


if __name__ == "__main__":
    main()
