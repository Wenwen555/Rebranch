"""Paired, no-LLM checks of task necessity and the legacy online probe."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from rebranch.api.primitives import PHYSCAP_EXTRA_PRIMITIVES
from rebranch.baselines import build_safe_probe_source
from rebranch.baselines.physcap_agent import PhysCaPEpisodeWrapper, _augment_probe_with_physcap
from rebranch.eval.runner import load_task_backend
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.sandbox import run_policy

OUT = Path("outputs/grounding_audit")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    # Freeze one action across all hidden modes: no latent parameter access.
    for task in ("t4_cabinet", "t5_drawer"):
        backend = load_task_backend(task)
        source = backend.draft_policy.replace("GRIP_FORCE = 14.0", "GRIP_FORCE = 18.0").replace("GRIP_FORCE = 15.0", "GRIP_FORCE = 18.0")
        old = "api.move_to([h[0] - PULL_DIST, h[1], h[2]], speed=PULL_SPEED)"
        assert old in source
        source = source.replace(old, 'api.move_compliant([h[0] - 0.125, h[1], h[2]], free_axes="yzXYZ", speed=0.04)')
        (OUT / f"{task}_fixed_compliant.py").write_text(source)
        for split, cases in (("hidden", backend.hidden_test_set(n=50)), ("boundary", backend.ood_test_set(n=10))):
            with Evaluator(backend, workers=8) as ev:
                traces = ev.run_many(source, cases, seed=0)
            (OUT / f"{task}_{split}_traces.jsonl").write_text("".join(json.dumps(asdict(t)) + "\n" for t in traces))
            row = {"task": task, "policy": "fixed_compliant", "split": split, "n": len(traces), "successes": sum(t.success for t in traces), "failures": dict(Counter(t.failure for t in traces if not t.success)), "by_mode": {mode: {"n": sum(backend.latent_mode(t.theta) == mode for t in traces), "successes": sum(t.success and backend.latent_mode(t.theta) == mode for t in traces)} for mode in sorted({backend.latent_mode(t.theta) for t in traces})}}
            summary.append(row)
            print(json.dumps(row), flush=True)
    for task in ("t2_cup", "t4_cabinet"):
        backend = load_task_backend(task)
        probe = _augment_probe_with_physcap(build_safe_probe_source(backend.draft_policy))
        (OUT / f"{task}_legacy_probe.py").write_text(probe)
        for i, theta in enumerate(backend.hidden_test_set(n=8)):
            ep = PhysCaPEpisodeWrapper(backend.make_episode(theta, seed=i))
            trace = run_policy(probe, ep, theta, seed=i, extra_primitives=PHYSCAP_EXTRA_PRIMITIVES)
            with (OUT / "probe_traces.jsonl").open("a") as fh:
                fh.write(json.dumps({"task": task, **asdict(trace)}) + "\n")
            row = {"task": task, "mode": backend.latent_mode(theta), "seed": i, "error": trace.error, "readings": {c.name: c.result for c in trace.calls if c.name in PHYSCAP_EXTRA_PRIMITIVES}}
            summary.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
