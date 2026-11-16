"""Frozen simple controls on paired benchmark cases and fresh parameter draws."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from rebranch.eval.runner import load_task_backend
from rebranch.exec.evaluator import Evaluator

OUT = Path("outputs/grounding_audit")


def main():
    summaries = []
    for task in ("t2_cup", "t4_cabinet", "t5_drawer"):
        backend = load_task_backend(task)
        if task == "t2_cup":
            policies = {f"fixed_force_{force}": backend.draft_policy.replace("GRIP_FORCE = 3.0", f"GRIP_FORCE = {force}.0") for force in (3, 6, 12)}
            policies["width_feedback"] = backend.draft_policy.replace(
                "    api.close_gripper(force=GRIP_FORCE)\n",
                '    api.close_gripper(force=2.5)\n    width = api.get_gripper_state()["width"]\n    if width > 0.044:\n        api.close_gripper(force=12.0)\n',
            )
        else:
            policies = {"fixed_compliant": (OUT / f"{task}_fixed_compliant.py").read_text()}
        for label, source in policies.items():
            (OUT / f"{task}_{label}.py").write_text(source)
            splits = [("hidden", backend.hidden_test_set(n=50), 0), ("hidden", backend.hidden_test_set(n=50), 1), ("boundary", backend.ood_test_set(n=10), 500)]
            if label in ("fixed_compliant", "width_feedback"):
                splits.append(("fresh", backend.hidden_test_set(seed=12026, n=100), 700))
            for split, cases, seed in splits:
                with Evaluator(backend, workers=8) as ev:
                    traces = ev.run_many(source, cases, seed=seed * 1000)
                (OUT / f"{task}_{label}_{split}_seed{seed}.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
                row = {"task": task, "policy": label, "split": split, "seed": seed, "n": len(traces), "successes": sum(t.success for t in traces), "failures": dict(Counter(t.failure for t in traces if not t.success)), "safety_violations": sum(t.safety_violation is not None for t in traces), "by_mode": {mode: {"n": sum(backend.latent_mode(t.theta) == mode for t in traces), "successes": sum(t.success and backend.latent_mode(t.theta) == mode for t in traces)} for mode in sorted({backend.latent_mode(t.theta) for t in traces})}}
                summaries.append(row)
                print(json.dumps(row), flush=True)
    (OUT / "controls_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
