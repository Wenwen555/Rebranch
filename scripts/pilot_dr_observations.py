"""Second repair with complete SDK traces from the same training batch."""

from collections import Counter
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np

from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.eval.runner import load_task_backend
from rebranch.exec.evaluator import Evaluator
from rebranch.llm.client import CodexCLIClient, DiskCacheLLM, extract_python_code
from rebranch.llm.prompts import SYSTEM_PROMPT
from scripts.pilot_dr_feedback import RecordedLLM

OUT = Path("outputs/grounding_audit/dr_observations")


def main(concise=False):
    out = OUT.with_name("dr_observations_concise") if concise else OUT
    task = load_task_backend("t2_cup")
    summaries = []
    for seed in (0, 1):
        directory = out / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        cases = [task.param_space.sample(rng) for _ in range(24)]
        rejected = Path(f"outputs/grounding_audit/cached_dr/seed{seed}_candidate0.py").read_text()
        evidence = []
        training = {}
        for label, source, noise_seed in [("Current policy", task.draft_policy, seed), ("Rejected attempt", rejected, seed + 100)]:
            with Evaluator(task, workers=8) as ev:
                traces = ev.run_many(source, cases, seed=noise_seed)
            training[label] = traces
            (directory / ("current.jsonl" if label == "Current policy" else "rejected.jsonl")).write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
            seen = set()
            for t in traces:
                outcome = "success" if t.success else t.failure or t.error
                if outcome in seen:
                    continue
                seen.add(outcome)
                evidence.append(label + ":\n" + summarize_trace_for_llm(t, max_calls=len(t.calls)))
        old_prompt = Path(f"outputs/grounding_audit/dr_feedback/seed{seed}/call2.prompt.txt").read_text()
        prompt = old_prompt.removeprefix(SYSTEM_PROMPT + "\n\n")
        prompt += "\n\nComplete SDK observations from one training trial per observed outcome, in execution order. These include early measurements omitted by the summaries above:\n" + "\n\n".join(evidence)
        if concise:
            prompt += (
                "\n\nWrite a short, task-directed policy. Use the supplied observations to choose actions. "
                "Add only checks required by the stated task constraints; omit speculative workspace checks, "
                "repeated sensor reads without intervening actions, and speculative abort/recovery paths."
            )
        cache = DiskCacheLLM(CodexCLIClient(), directory / "cache")
        response = RecordedLLM(cache, directory).complete(prompt, system=SYSTEM_PROMPT)
        candidate = extract_python_code(response.text)
        (directory / "candidate.py").write_text(candidate)
        with Evaluator(task, workers=8) as ev:
            candidate_traces = ev.run_many(candidate, cases, seed=seed + 200)
        (directory / "candidate_training.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in candidate_traces))
        old = training["Current policy"]
        old_quality = (sum(t.success for t in old), sum(t.score for t in old))
        new_quality = (sum(t.success for t in candidate_traces), sum(t.score for t in candidate_traces))
        accepted = new_quality > old_quality
        final = candidate if accepted else task.draft_policy
        (directory / "final.py").write_text(final)
        record = {"seed": seed, "accepted": accepted, "candidate_training_successes": new_quality[0], "training_n": 24, "llm": cache.stats, "logical_total_proposals": 2, "replayed_training_rollouts": 48, "new_candidate_training_rollouts": 24, "evaluation": []}
        for split, param_seed, n, noise_seed in [("hidden", 2026, 50, seed * 1000), ("fresh", 12026, 100, 700000)]:
            with Evaluator(task, workers=8) as ev:
                traces = ev.run_many(final, task.hidden_test_set(seed=param_seed, n=n), seed=noise_seed)
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
            record["evaluation"].append({"split": split, "n": n, "successes": sum(t.success for t in traces), "failures": dict(Counter(t.failure for t in traces if not t.success)), "safety_violations": sum(bool(t.safety_violation) for t in traces)})
        summaries.append(record)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concise", action="store_true")
    main(concise=parser.parse_args().concise)
