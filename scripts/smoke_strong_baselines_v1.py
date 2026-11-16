"""Bounded real-MuJoCo smoke for the strong baseline runner integration."""

from __future__ import annotations

import json
from pathlib import Path

from rebranch.eval.runner import run_single_method


class NoModelCalls:
    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected experiment-model access: {name}")


def main() -> None:
    root = Path("outputs/repair_validation/strong_baselines_v1")
    root.mkdir(parents=True, exist_ok=True)
    llm = NoModelCalls()
    runs = [
        ("t2_cup", "parameter_only", 96, 1, 0, {"training_cases_per_restart": 2, "optimizer_restart_rollouts": 48}),
        ("t4_cabinet", "fixed_compliant", 2000, 1, 0, {}),
        ("t5_drawer", "fixed_compliant", 2000, 1, 0, {}),
        ("t4_cabinet", "compliant_parameter_only", 0, 0, 0, {}),
    ]
    summary = {"model_calls": 0, "actual_rollouts_total": 0, "jobs": []}
    for task, method, cap, n_hidden, n_ood, kwargs in runs:
        out = run_single_method(
            task_name=task,
            method=method,
            llm=llm,  # This object raises on any attempted model access.
            seed=0,
            workers=1,
            max_rollouts=cap,
            n_hidden=n_hidden,
            n_ood=n_ood,
            output_dir=root / "smoke_runs" / f"{task}__{method}",
            **kwargs,
        )
        res = out["result"]
        test_rollouts = sum(int(m.get("n_episodes", 0)) for m in out["metrics"].values())
        actual = int(res.n_rollouts) + test_rollouts
        summary["actual_rollouts_total"] += actual
        summary["jobs"].append({
            "task": task,
            "method": method,
            "dev_rollouts": int(res.n_rollouts),
            "test_rollouts": test_rollouts,
            "actual_rollouts": actual,
            "dev_budget": cap,
            "dev_llm_calls": int(res.dev_llm_calls),
            "rounds": res.round_logs,
            "extra": res.extra,
            "test_metrics": out["metrics"],
        })
    summary["model_calls"] = llm.calls
    if summary["actual_rollouts_total"] > 100:
        raise AssertionError(f"smoke exceeded 100 rollouts: {summary['actual_rollouts_total']}")
    if llm.calls:
        raise AssertionError(f"unexpected experiment-model calls: {llm.calls}")
    (root / "smoke_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_calls": summary["model_calls"],
        "actual_rollouts_total": summary["actual_rollouts_total"],
        "jobs": [
            {"task": j["task"], "method": j["method"], "dev_rollouts": j["dev_rollouts"],
             "test_rollouts": j["test_rollouts"], "success_count": j["extra"].get("validation", {}).get("final_success_count")}
            for j in summary["jobs"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
