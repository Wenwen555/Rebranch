"""Small no-LLM regression run on the six saved stage2 candidate programs."""

from __future__ import annotations

import ast
import argparse
import json
from collections import Counter
from pathlib import Path

from rebranch.baselines import extract_float_constants, instantiate_float_constants
from rebranch.core.lifting import instantiate, lift_constants
from rebranch.core.retune import Retuner
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.trace import CallRecord, Trace
from rebranch.core.development import AdmissionGate
from rebranch.tasks import make_task


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "outputs/repair_validation/stage2"
OUT = ROOT / "outputs/repair_validation/stage1d"
JOBS = {
    "t2_cup__uniform_dr__seed0": "t2",
    "t4_cabinet__uniform_dr__seed0": "t4",
    "t5_drawer__uniform_dr__seed0": "t5",
}


def _summary(traces):
    return {
        "n": len(traces),
        "success": sum(t.success for t in traces),
        "task_failures": dict(Counter(t.failure or "none" for t in traces)),
        "execution_errors": dict(Counter(t.error or "none" for t in traces)),
        "max_calls": max((len(t.calls) for t in traces), default=0),
    }


def _assignment_types(source):
    tree = ast.parse(source)
    out = {}
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            continue
        try:
            value = ast.literal_eval(stmt.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[stmt.targets[0].id] = (value, type(value).__name__)
    return out


def _trace_from_json(data):
    row = dict(data)
    row["calls"] = [CallRecord(**call) for call in row.get("calls", [])]
    return Trace(**row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-only", action="store_true",
                        help="reuse completed candidate results and refresh only the saved-trace feedback check")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / "real_candidate_offline.json"
    result = (json.loads(previous.read_text()) if args.feedback_only and previous.exists() else
              {"llm_requests": 0, "case_source": "saved development_traces.jsonl; first two records per job",
               "jobs": {}})
    for job, task_name in ({} if args.feedback_only else JOBS).items():
        run_dir = STAGE2 / "runs" / job
        cases = []
        with (run_dir / "development_traces.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                tr = row["trace"]
                cases.append((tr["theta"], int(tr["seed"])))
                if len(cases) == 2:
                    break
        if len(cases) != 2:
            raise RuntimeError(f"expected two stored development cases for {job}")
        backend = make_task(task_name)
        job_result = {}
        for req in ("01", "02"):
            source = (run_dir / "requests" / req / "candidate.py").read_text()
            specs = lift_constants(source)
            ints = [c for c in specs if c.kind == "continuous" and c.is_int]
            if not ints:
                raise AssertionError(f"{job}/{req}: expected at least one lifted integer literal")
            # Check every lifted integer's two legal bounds and fixed initial value.
            integer_checks = {}
            fixed = {c.name: c.initial for c in specs}
            for spec in ints:
                values = [Retuner._to_values([spec], [u], {}, {})[spec.name] for u in (0.0, 0.5, 1.0)]
                assert all(type(v) is int for v in values)
                assert all(__import__("math").ceil(spec.lo) <= v <= __import__("math").floor(spec.hi) for v in values)
                assert type(fixed[spec.name]) is int
                integer_checks[spec.name] = {"initial": spec.initial, "samples": values, "bounds": [spec.lo, spec.hi]}
            rewritten = instantiate(source, fixed)
            rewritten_types = _assignment_types(rewritten)
            for spec in ints:
                assert rewritten_types[spec.name][1] == "int"
            for spec in specs:
                if spec.kind == "continuous" and not spec.is_int:
                    assert type(fixed[spec.name]) is float
                    assert _assignment_types(rewritten)[spec.name][1] == "float"

            # Exercise the shared Twin/OneShot constant-instantiation helper too.
            baseline_values = {name: value for name, value, _, _ in extract_float_constants(source)}
            baseline_rewrite = instantiate_float_constants(source, baseline_values)
            baseline_types = _assignment_types(baseline_rewrite)
            original_types = _assignment_types(source)
            for name, value in baseline_values.items():
                expected = original_types[name][1]
                assert baseline_types[name][1] == expected, (job, req, name, baseline_types[name])

            with Evaluator(backend, workers=1, budget_limit=10) as evaluator:
                original_traces = evaluator.run_many(source, cases, seed=10000)
                fixed_traces = evaluator.run_many(rewritten, cases, seed=10000)
                focus = next((c for c in ints if c.name == "MAX_STEPS"), ints[0])
                retuner = Retuner(evaluator, max_iters=1, popsize=2, seed=177,
                                  max_rollouts=6, budget_reserve=0)
                optimized = retuner.optimize(source, [focus], cases)
                retuned_traces = optimized.traces
                assert all("TypeError" not in (t.error or "") for t in original_traces + fixed_traces + retuned_traces)
                assert all(type(v) is int for v in optimized.best_values.values())
                assert _assignment_types(optimized.best_source)[focus.name][1] == "int"

            job_result[req] = {
                "candidate_sha256": __import__("hashlib").sha256(source.encode()).hexdigest(),
                "n_lifted_integer_constants": len(ints),
                "integer_boundary_and_fixed_checks": integer_checks,
                "original_candidate_two_case_eval": _summary(original_traces),
                "fixed_value_rewrite_two_case_eval": _summary(fixed_traces),
                "short_retuner": {"parameter": focus.name, "n_evals": optimized.n_evals,
                                  "best_value": optimized.best_values[focus.name],
                                  "best_literal_type": _assignment_types(optimized.best_source)[focus.name][1],
                                  "traces": _summary(retuned_traces)},
                "twin_oneshot_instantiator_integer_literal_type": baseline_types.get(focus.name),
            }
        result["jobs"][job] = job_result
    # Feed actual saved 200-call and TypeError traces through the bounded admission path.
    raw_traces = []
    trace_path = STAGE2 / "runs/t2_cup__uniform_dr__seed0/development_traces.jsonl"
    selected = {}
    with trace_path.open() as f:
        for line in f:
            item = json.loads(line)["trace"]
            error = item.get("error") or ""
            if "TypeError" in error and "type_error" not in selected:
                selected["type_error"] = item
            if "max_calls (200)" in error and "call_budget" not in selected:
                selected["call_budget"] = item
            if len(selected) == 2:
                break
    if set(selected) != {"type_error", "call_budget"}:
        raise AssertionError("saved trace corpus lacks TypeError or max_calls example")
    raw_traces = [_trace_from_json(selected["type_error"]), _trace_from_json(selected["call_budget"])]

    class SavedTraceEvaluator:
        def run_many(self, source, cases, seed):
            if source == "incumbent":
                return [Trace(theta={}, seed=i, success=True, score=1.0) for i in range(24)]
            return [raw_traces[i % 2] for i in range(24)]

    gate = AdmissionGate(SavedTraceEvaluator(), [{} for _ in range(24)], "incumbent")
    real_candidate_path = STAGE2 / "runs/t2_cup__uniform_dr__seed0/requests/01/candidate.py"
    real_candidate = real_candidate_path.read_text().strip()
    gate.consider(real_candidate)
    feedback = gate.feedback
    assert "TypeError" in feedback and "max_calls (200) exceeded" in feedback
    assert "All 24 admission cases retained" in feedback
    assert len(feedback) < 30000 and feedback.count("Recent SDK calls (showing 8 of 200)") <= 3
    result["real_long_trace_feedback"] = {
        "source": str(trace_path.relative_to(ROOT)),
        "real_typeerror_calls": len(raw_traces[0].calls),
        "real_call_budget_calls": len(raw_traces[1].calls),
        "feedback_chars": len(feedback),
        "feedback_case_details": feedback.count("Admission case "),
        "candidate_source_chars": len(real_candidate),
        "candidate_source_occurrences": feedback.count(real_candidate),
        "contains_typeerror": "TypeError" in feedback,
        "contains_call_budget": "max_calls (200) exceeded" in feedback,
        "all_24_counts_retained": "All 24 admission cases retained" in feedback,
        "llm_requests": 0,
    }
    (OUT / "real_candidate_offline.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
