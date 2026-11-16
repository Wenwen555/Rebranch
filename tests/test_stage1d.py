from __future__ import annotations

import ast

from rebranch.baselines import instantiate_float_constants
from rebranch.core.development import AdmissionGate, DEVELOPMENT_PROTOCOL_VERSION
from rebranch.core.lifting import lift_constants
from rebranch.core.retune import Retuner
from rebranch.core.branch import format_conflict_summary, format_trace_feedback
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.exec.trace import CallRecord, Trace
from rebranch.api.primitives import API_DOC
from rebranch.llm.prompts import SHARED_PROMPT_VERSION, SYSTEM_PROMPT


def test_integer_literal_type_survives_lifting_quantization_and_instantiation():
    source = "N = 3  # [1.0, 5.0]\ndef run(api):\n    for _ in range(N):\n        pass\n"
    spec, = lift_constants(source)
    assert spec.kind == "continuous" and spec.is_int and spec.initial == 3
    values = Retuner._to_values([spec], [0.0], {}, {})
    assert values["N"] == 1 and isinstance(values["N"], int)
    values = Retuner._to_values([spec], [1.0], {}, {})
    assert values["N"] == 5 and isinstance(values["N"], int)
    values = Retuner._to_values([spec], [0.5], {}, {})
    assert values["N"] == 3 and isinstance(values["N"], int)
    tuned = ast.parse(__import__("rebranch.core.lifting", fromlist=["instantiate"]).instantiate(source, values))
    assert isinstance(tuned.body[0].value, ast.Constant) and type(tuned.body[0].value.value) is int

    negative, = lift_constants("N = -3  # [-5.0, -1.0]\n")
    assert negative.is_int and negative.initial == -3
    assert Retuner._to_values([negative], [0.0], {}, {})["N"] == -5
    assert Retuner._to_values([negative], [1.0], {}, {})["N"] == -1

    floating, = lift_constants("N = 3.0  # [1.0, 5.0]\n")
    assert not floating.is_int and type(floating.initial) is float
    assert type(Retuner._to_values([floating], [0.5], {}, {})["N"]) is float


def test_baseline_constant_instantiation_keeps_integer_range_counter():
    source = ("MAX_STEPS = 160  # [60, 240]\n"
              "NEGATIVE_COUNT = -3  # [-5.0, -1.0]\n"
              "FLOAT_GAIN = 1.23456  # [0.0, 2.0]\n"
              "for _ in range(MAX_STEPS):\n    pass\n")
    tuned = instantiate_float_constants(source, {"MAX_STEPS": 173.7,
                                                  "NEGATIVE_COUNT": -3.7,
                                                  "FLOAT_GAIN": 1.23456})
    tree = ast.parse(tuned)
    values = {stmt.targets[0].id: ast.literal_eval(stmt.value) for stmt in tree.body[:3]}
    assert type(values["MAX_STEPS"]) is int and values["MAX_STEPS"] == 174
    assert type(values["NEGATIVE_COUNT"]) is int and values["NEGATIVE_COUNT"] == -4
    assert type(values["FLOAT_GAIN"]) is float and values["FLOAT_GAIN"] == 1.2346
    compile(tuned, "<policy>", "exec")
    assert "MAX_STEPS = 174" in tuned


class _AdmissionEvaluator:
    def __init__(self):
        self.calls = 0

    def run_many(self, source, cases, seed):
        self.calls += 1
        if source == "incumbent":
            return [Trace(theta={}, seed=i, success=True, score=1.0) for i in range(24)]
        result = []
        for i in range(24):
            failed = i in (2, 4, 6, 8)
            error = "TypeError: float object cannot be interpreted as an integer" if i == 4 else (
                "API call budget exceeded (200)" if i == 6 else None
            )
            calls = [CallRecord(j, "get_object_pose", 7, 1, ("handle",), {"frame": "world"},
                                {"pos": [0.3643, 0.0, 0.095]},
                                {"ee_x": 0.2467, "fx": 0.6, "fy": -0.02, "fz": 0.5,
                                 "tx": 0.0, "ty": 0.23, "tz": 0.0,
                                 "grip_width": 0.0002, "grip_effort": 18.95,
                                 "grip_grasped": 0.0}, float(j))
                     for j in range(200)]
            result.append(Trace(theta={}, seed=i, calls=calls, success=not failed,
                                score=0.0 if failed else 0.5,
                                failure="not_lifted" if i == 2 else None,
                                error=error))
        return result


def test_admission_feedback_keeps_counts_and_bounds_sdk_details():
    evaluator = _AdmissionEvaluator()
    gate = AdmissionGate(evaluator, [{} for _ in range(24)], "incumbent")
    candidate = "# UNIQUE_CANDIDATE_SOURCE\ndef run(api):\n    pass\n"
    accepted, _ = gate.consider(candidate)
    assert not accepted
    feedback = gate.feedback
    assert "old=(24, 1.0), new=(20, 0.4166666666666667)" in feedback
    assert "success regressions at admission indices [2, 4, 6, 8]" in feedback
    assert "All 24 admission cases retained" in feedback
    assert "task_failure=not_lifted; execution_error=none: 1" in feedback
    assert "Execution error: TypeError: float object cannot be interpreted as an integer" in feedback
    assert "Execution error: API call budget exceeded (200)" in feedback
    assert feedback.count("Admission case ") <= 3
    assert feedback.count("Recent SDK calls (showing 8 of 200)") <= 3
    assert "get_object_pose" in feedback and "'handle'" in feedback
    assert "0.3643" in feedback and "grip_effort" in feedback
    assert feedback.count("UNIQUE_CANDIDATE_SOURCE") == 1
    assert len(feedback) < 30_000
    assert "theta" not in feedback


def test_shared_documentation_declares_call_budget_and_protocol_versions():
    assert "200" in API_DOC and "getter" in API_DOC.lower()
    assert SHARED_PROMPT_VERSION == "stage1d-common-control-v2"
    assert "documented per-policy API-call budget" in SYSTEM_PROMPT
    assert "speculative preflight" in SYSTEM_PROMPT
    assert DEVELOPMENT_PROTOCOL_VERSION == "repair-development-v4-bounded-feedback"


def test_trace_summary_shows_failure_and_error_separately():
    trace = Trace(theta={}, seed=0, success=False, score=-0.2,
                  failure="insufficient_pull", error="TypeError: float object cannot be interpreted as an integer")
    summary = summarize_trace_for_llm(trace)
    assert "Task failure: insufficient_pull" in summary
    assert "Execution error: TypeError: float object cannot be interpreted as an integer" in summary
    budget_trace = Trace(theta={}, seed=1, success=False, score=-0.3,
                         error="max_calls (200) exceeded")
    branch_summary = format_trace_feedback([trace, budget_trace])
    conflict_summary = format_conflict_summary([trace, budget_trace], [])
    for branch_path in (branch_summary, conflict_summary):
        assert "insufficient_pull" in branch_path
        assert "TypeError: float object cannot be interpreted as an integer" in branch_path
        assert "max_calls (200) exceeded" in branch_path
