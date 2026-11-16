"""Compare ordinary and branch SDK summaries on the saved real T5 failure."""

from __future__ import annotations

import json
from pathlib import Path

from rebranch.core.branch import format_trace_feedback
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.exec.trace import CallRecord, Trace


SOURCE_SHA = "cbed3416ca1e818e0dfd75c31959790ce7e41eb0290df5954b2c1b4b70a65654"
TRACE_PATH = Path("outputs/repair_validation/stage2d/runs/t5_drawer__uniform_dr__seed0/development_traces.jsonl")
OUT_PATH = Path("outputs/repair_validation/stage1e/real_t5_summary_check.json")


def main() -> None:
    selected = None
    for line in TRACE_PATH.open():
        row = json.loads(line)
        if row["source_sha256"] != SOURCE_SHA:
            continue
        raw = row["trace"]
        if not any(c["name"] == "get_object_pose"
                   and abs(c["result"]["pos"][0] - 0.36429440145537395) < 1e-8
                   for c in raw["calls"]):
            continue
        selected = raw
        break
    assert selected is not None, "saved T5 candidate01 evidence trace not found"
    trace = Trace(
        theta={}, seed=selected["seed"],
        calls=[CallRecord(**call) for call in selected["calls"]],
        success=selected["success"], score=selected["score"],
        failure=selected.get("failure"), error=selected.get("error"),
        safety_violation=selected.get("safety_violation"),
        sim_time=selected.get("sim_time", 0.0),
    )
    ordinary = summarize_trace_for_llm(trace, max_calls=8)
    branch = format_trace_feedback([trace], max_calls_per_trace=8)
    required = ("get_object_pose", "0.3643", "0.2467", "grip_effort",
                "grip_width", "grip_grasped", "fx", "fy", "fz", "tx", "ty", "tz")
    for name, summary in (("ordinary", ordinary), ("branch", branch)):
        missing = [field for field in required if field not in summary]
        assert not missing, f"{name} summary missing SDK fields: {missing}"
        assert "theta" not in summary
        assert "insufficient_pull" in summary
    assert "'handle'" in ordinary and "'handle'" in branch
    assert "18.9500" in ordinary and "18.9500" in branch
    assert len(trace.calls[-8:]) == 8
    result = {
        "source_sha256": SOURCE_SHA,
        "seed": trace.seed,
        "calls_in_trace": len(trace.calls),
        "failure": trace.failure,
        "ee_x": trace.calls[-1].obs.get("ee_x"),
        "handle_x": next(c.result["pos"][0] for c in reversed(trace.calls)
                          if c.name == "get_object_pose" and c.result),
        "last_grip_effort": trace.calls[-1].obs.get("grip_effort"),
        "last_grip_width": trace.calls[-1].obs.get("grip_width"),
        "last_grip_grasped": trace.calls[-1].obs.get("grip_grasped"),
        "ordinary_chars": len(ordinary),
        "branch_chars": len(branch),
        "ordinary_summary": ordinary,
        "branch_summary": branch,
        "theta_included": False,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if "summary" not in k}, indent=2))


if __name__ == "__main__":
    main()
