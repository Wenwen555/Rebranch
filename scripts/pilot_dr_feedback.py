"""Matched-budget T2 pilot: pass rejected-program feedback to the next repair."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from rebranch.baselines.uniform_dr import run_uniform_dr
from rebranch.eval.runner import load_task_backend
from rebranch.exec.evaluator import Evaluator
from rebranch.llm.client import CodexCLIClient, DiskCacheLLM

OUT = Path("outputs/grounding_audit/dr_feedback")


class RecordedLLM:
    def __init__(self, cache, directory):
        self.cache = cache
        self.directory = directory
        self.calls = 0

    def complete(self, prompt, *, system="", temperature=0.2):
        self.calls += 1
        prefix = self.directory / f"call{self.calls}"
        prefix.with_suffix(".prompt.txt").write_text(system + "\n\n" + prompt)
        response = self.cache.complete(prompt, system=system, temperature=temperature)
        prefix.with_suffix(".response.json").write_text(json.dumps(asdict(response), indent=2))
        print(json.dumps({"directory": str(self.directory), "call": self.calls, "cached": response.cached}), flush=True)
        return response


def main():
    task = load_task_backend("t2_cup")
    rows = [json.loads(x) for x in Path("outputs/m4/runs.jsonl").read_text().splitlines()]
    old = json.loads(Path("outputs/grounding_audit/cached_dr/summary.json").read_text())
    summaries = []
    for seed in (0, 1):
        directory = OUT / f"seed{seed}"
        cache_dir = directory / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for key in {r["response_key"] for r in old if r["seed"] == seed}:
            shutil.copyfile(Path(".cache/rebranch_llm/m4") / f"{key}.json", cache_dir / f"{key}.json")
        cache = DiskCacheLLM(CodexCLIClient(), cache_dir)
        llm = RecordedLLM(cache, directory)
        row = next(r for r in rows if (r["task"], r["method"], r["split"], r["seed"]) == ("t2_cup", "uniform_dr", "hidden", seed))
        result = run_uniform_dr(task, llm, seed=seed, workers=8, **row["budget"])
        (directory / "result.json").write_text(json.dumps(asdict(result), indent=2))
        (directory / "final.py").write_text(result.policy_source)
        for snapshot in result.program_snapshots:
            (directory / f"round{snapshot['round']}.py").write_text(snapshot["source"])
        stats = {"seed": seed, "budget": row["budget"], "n_rollouts": result.n_rollouts, "llm": cache.stats, "evaluation": []}
        for split, param_seed, n, noise_seed in [("hidden", 2026, 50, seed * 1000), ("fresh", 12026, 100, 700000)]:
            with Evaluator(task, workers=8) as ev:
                traces = ev.run_many(result.policy_source, task.hidden_test_set(seed=param_seed, n=n), seed=noise_seed)
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
            stats["evaluation"].append({"split": split, "parameter_seed": param_seed, "noise_seed": noise_seed, "n": n, "successes": sum(t.success for t in traces), "failures": dict(Counter(t.failure for t in traces if not t.success)), "safety_violations": sum(bool(t.safety_violation) for t in traces)})
        summaries.append(stats)
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
