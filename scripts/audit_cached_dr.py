"""Recover rejected DR candidates and check mode-specific failures without LLM calls."""

from collections import Counter
from dataclasses import asdict
import json
import importlib.util
from pathlib import Path

from rebranch.eval.runner import load_task_backend
from rebranch.exec.evaluator import Evaluator
from rebranch.llm.client import extract_python_code
from scripts.audit_cached_physcap import Replay

OUT = Path("outputs/grounding_audit/cached_dr")


def main():
    spec = importlib.util.spec_from_file_location("uniform_dr_legacy", OUT / "uniform_dr_legacy.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    rows = [json.loads(x) for x in Path("outputs/m4/runs.jsonl").read_text().splitlines()]
    task = load_task_backend("t2_cup")
    summary = []
    for seed in (0, 1):
        row = next(r for r in rows if (r["task"], r["method"], r["split"], r["seed"]) == ("t2_cup", "uniform_dr", "hidden", seed))
        replay = Replay(OUT)
        result = legacy.run_uniform_dr(task, replay, seed=seed, workers=8, **row["budget"])
        (OUT / f"seed{seed}_rounds.json").write_text(json.dumps(result.round_logs, indent=2))
        for idx, response in enumerate(replay.responses):
            source = extract_python_code(response)
            (OUT / f"seed{seed}_candidate{idx}.py").write_text(source)
            with Evaluator(task, workers=8) as ev:
                traces = ev.run_many(source, task.hidden_test_set(n=50), seed=seed * 1000)
            (OUT / f"seed{seed}_candidate{idx}_traces.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
            record = {"seed": seed, "candidate": idx, "successes": sum(t.success for t in traces), "n": len(traces), "failure_counts": dict(Counter(t.failure for t in traces if not t.success)), "mode_successes": {mode: sum(t.success and task.latent_mode(t.theta) == mode for t in traces) for mode in ("empty", "sand")}, "response_key": replay.keys[idx]}
            summary.append(record)
            print(json.dumps(record), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
