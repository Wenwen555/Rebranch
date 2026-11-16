"""Replay legacy PhysCaP from saved responses; never issue an LLM request."""

from dataclasses import asdict
import json
from pathlib import Path

from rebranch.baselines.physcap_agent import run_physcap
from rebranch.eval.runner import load_task_backend
from rebranch.llm.client import CodexCLIClient, DiskCacheLLM, LLMResponse

OUT = Path("outputs/grounding_audit/cached_physcap")


class Replay:
    def __init__(self, output_dir=OUT):
        self.cache = DiskCacheLLM(CodexCLIClient(), ".cache/rebranch_llm/m4")
        self.keys = []
        self.responses = []
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, prompt, *, system="", temperature=0.2):
        key = self.cache._cache_key(prompt, system, temperature)
        data = json.loads((self.cache.cache_dir / f"{key}.json").read_text())
        self.keys.append(key)
        self.responses.append(data["text"])
        (self.output_dir / f"{key}_prompt.txt").write_text(prompt)
        (self.output_dir / f"{key}_response.txt").write_text(data["text"])
        return LLMResponse(text=data["text"], prompt_tokens=data["prompt_tokens"], completion_tokens=data["completion_tokens"], cached=True, wall_s=0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("t2_cup", "t4_cabinet"):
        task = load_task_backend(name)
        replay = Replay()
        agent = run_physcap(task, replay).agent
        for i, theta in enumerate(task.hidden_test_set(n=4)):
            trace, meta = agent.evaluate_episode(task, theta, seed=i)
            (OUT / f"{name}_{i}_trace.json").write_text(json.dumps(asdict(trace), indent=2))
            (OUT / f"{name}_{i}_policy.py").write_text(meta["policy_source"])
            row = {"task": name, "seed": i, "success": trace.success, "error": trace.error, "failure": trace.failure, "calls": len(trace.calls), "last_calls": [c.name for c in trace.calls[-5:]], "last_lines": trace.lines[-10:], "response_keys": list(replay.keys)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
