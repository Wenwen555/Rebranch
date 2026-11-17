from __future__ import annotations

import json

import pytest

from rebranch.baselines.agentic_genplan import (
    AgenticGenPlanAgent,
    _safe_analysis,
    _run_agentic_policy,
    _run_agentic_policy_source_check,
    run_agentic_genplan,
)
from rebranch.exec.sandbox import StaticCheckError
from rebranch.llm.client import MockLLMClient
from rebranch.sim.toy_backend import ToyRetuneTask


POLICY = '''import numpy as np
GAIN = 2.0
def helper(x):
    return float(np.sqrt(x)) * GAIN
def run(api):
    print(helper(4))
    api.zero_ee_wrench()
'''
TEST = '''def run(api):
    api.get_ee_pose()
'''


def _action(tool: str, **kwargs) -> str:
    """Encode the canonical tool request envelope used by the model interface."""
    return json.dumps({"tool": tool, "arguments": kwargs})


def test_agentic_tool_loop_has_schema_history_tests_and_freeze():
    outputs = [
        _action("inspect_task"),
        _action("write_test", source=TEST),
        _action("run_test", case_indices=[1]),
        _action("inspect_run", run_id=0),
        _action("run_analysis", script='import numpy as np\ndef analyze(data):\n    return {"n": len(data), "mean": float(np.mean([x["score"] for x in data]))}') ,
        _action("write_policy", source=POLICY),
        _action("run_experiment", source_kind="policy", case_indices=[2]),
        _action("freeze_policy"),
    ]
    llm = MockLLMClient(outputs)
    result = run_agentic_genplan(ToyRetuneTask(), llm, seed=4, max_rollouts=4,
                                 max_llm_calls=8, n_dev_cases=4)
    assert result.method == "agentic_genplan"
    assert result.extra["frozen"] is True
    assert result.extra["n_experiments"] == 2
    assert result.extra["development_runs"][0]["sdk_calls"][0]["name"] == "get_ee_pose"
    assert result.extra["development_runs"][1]["sdk_calls"][0]["name"] == "zero_ee_wrench"
    assert result.extra["analysis_scripts"][0]["result"]["n"] == 1
    assert "draft_policy" not in llm.history[0]["prompt"]
    assert '"write_test"' in llm.history[0]["prompt"]
    assert '"source_kind"' in llm.history[0]["prompt"]
    assert '"case_indices"' in llm.history[0]["prompt"]
    assert '"run_id"' in llm.history[0]["prompt"]
    assert '"script"' in llm.history[0]["prompt"]
    final_prompt = llm.history[-1]["prompt"]
    assert "inspect_task" in final_prompt and "get_ee_pose" in final_prompt
    assert "remaining_rollouts" in final_prompt and "remaining_llm_requests" in final_prompt
    assert result.dev_llm_calls == 8
    assert llm.stats["calls"] == 8
    assert result.policy_source == POLICY
    for request in llm.history:
        prompt = json.loads(request["prompt"])
        assert prompt["response_schema"]["required"] == ["tool", "arguments"]
        assert prompt["response_example"]["tool"] == "write_policy"
        assert "source" in prompt["response_example"]["arguments"]
        assert prompt["case_index_range"] == {"min_inclusive": 0, "max_inclusive": 3}
        assert "legal_case_indices" not in prompt
        assert "source" in prompt["tools"]["write_policy"]["arguments"]
    assert result.extra["tool_events"][5]["action"] == {
        "tool": "write_policy", "arguments": {"source": POLICY},
    }


def test_rollout_exhaustion_still_allows_analysis_and_explicit_freeze():
    llm = MockLLMClient([
        _action("inspect_task"),
        _action("write_policy", source=POLICY),
        _action("run_experiment", case_indices=[0]),
        _action("run_analysis", script='def analyze(data):\n    return len(data)'),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=1,
                                 max_llm_calls=5, n_dev_cases=2)
    assert result.extra["frozen"] is True
    assert result.extra["stop_reason"] == "explicit_freeze"
    assert result.n_rollouts == 1
    assert len(llm.history) == 5
    assert "\"remaining_rollouts\": 0" in llm.history[-1]["prompt"]


def test_agentic_executor_uses_shared_globals_for_imports_helpers_and_constants():
    backend = ToyRetuneTask()
    theta = backend.param_space.nominal()
    episode = backend.make_episode(theta, seed=0)
    trace = _run_agentic_policy(POLICY, episode, theta, seed=0, max_calls=5)
    assert trace.error is None
    assert trace.stdout.strip() == "4.0"
    assert [c.name for c in trace.calls] == ["zero_ee_wrench"]
    agent = AgenticGenPlanAgent(POLICY)
    trace2, metadata = agent.evaluate_episode(backend, theta, seed=1)
    assert trace2.error is None and metadata["runtime_llm_calls"] == 0


def test_function_local_import_and_helper_survive_static_import_check():
    source = chr(10).join([
        "def helper(x):", "    return x + 1", "def run(api):",
        "    import numpy as np", "    value = np.sqrt(helper(3))",
        "    if isinstance(value, np.floating):", "        api.zero_ee_wrench()", "",
    ])
    backend = ToyRetuneTask()
    theta = backend.param_space.nominal()
    trace = _run_agentic_policy(source, backend.make_episode(theta, seed=2), theta,
                                seed=2, max_calls=5)
    assert trace.error is None
    assert [call.name for call in trace.calls] == ["zero_ee_wrench"]


def test_analysis_normalizes_numpy_scalars_and_arrays_for_json():
    script = chr(10).join([
        "import numpy as np", "def analyze(data):",
        "    return {'scalar': np.float64(1.25), 'array': np.array([1, 2])}", "",
    ])
    assert _safe_analysis(script, []) == {"scalar": 1.25, "array": [1, 2]}


def test_import_only_control_flow_compiles_and_source_inspection_modules_are_blocked():
    source = chr(10).join([
        "def run(api):", "    if True:", "        import numpy as np",
        "    api.get_ee_pose()", "",
    ])
    backend = ToyRetuneTask()
    theta = backend.param_space.nominal()
    trace = _run_agentic_policy(source, backend.make_episode(theta, seed=3), theta,
                                seed=3, max_calls=5)
    assert trace.error is None
    for module in ("builtins", "io", "inspect", "gc"):
        with pytest.raises(StaticCheckError):
            _run_agentic_policy_source_check(f"import {module}" + chr(10) + "def run(api):" + chr(10) + "    pass")


def test_invalid_case_indices_are_rejected_before_any_episode_runs():
    backend = ToyRetuneTask()
    opened = []
    original_make_episode = backend.make_episode
    def counted_make_episode(*args, **kwargs):
        opened.append(1)
        return original_make_episode(*args, **kwargs)
    backend.make_episode = counted_make_episode
    llm = MockLLMClient([
        _action("write_policy", source=POLICY),
        _action("run_experiment", case_indices=[0, 4]),
        _action("run_experiment", case_indices=[0]),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(backend, llm, max_rollouts=2,
                                 max_llm_calls=4, n_dev_cases=2)
    assert result.n_rollouts == 1
    assert len(opened) == 1
    history = result.extra["tool_events"]
    assert "out of range" in history[1]["tool_result"]["error"]


def test_external_llm_stop_iteration_stops_cleanly_after_policy_commit():
    class StopsAfterPolicy:
        def __init__(self):
            self.responses = iter([_action("write_policy", source=POLICY)])
        def complete(self, *args, **kwargs):
            return type("Response", (), {"text": next(self.responses),
                                          "prompt_tokens": 1,
                                          "completion_tokens": 1})()
    result = run_agentic_genplan(ToyRetuneTask(), StopsAfterPolicy(), max_rollouts=2,
                                 max_llm_calls=4, n_dev_cases=1)
    assert result.policy_source == POLICY
    assert result.dev_llm_calls == 1
    assert result.extra["stop_reason"] == "external_llm_budget_exhausted"
    assert result.extra["frozen"] is True


def test_analysis_receives_copy_and_cannot_mutate_audit_runs():
    llm = MockLLMClient([
        _action("write_policy", source=POLICY),
        _action("run_experiment", case_indices=[0]),
        _action("run_analysis", script=chr(10).join(["def analyze(data):", "    data.pop()", "    return len(data)"])),
        _action("run_analysis", script=chr(10).join(["def analyze(data):", "    return len(data)"])),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=2,
                                 max_llm_calls=5, n_dev_cases=1)
    assert result.extra["analysis_scripts"][0]["result"] == 0
    assert result.extra["analysis_scripts"][1]["result"] == 1
    assert len(result.extra["development_runs"]) == 1


def test_no_generated_policy_is_not_silently_replaced_by_draft():
    with pytest.raises(RuntimeError, match="no valid committed policy"):
        run_agentic_genplan(ToyRetuneTask(), MockLLMClient([_action("inspect_task")]),
                            max_rollouts=1, max_llm_calls=1, n_dev_cases=1)


def test_budget_stop_freezes_last_committed_policy_without_draft_fallback():
    llm = MockLLMClient([
        _action("inspect_task"),
        _action("write_policy", source=POLICY),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=5,
                                 max_llm_calls=2, n_dev_cases=2)
    assert result.extra["frozen"] is True
    assert result.extra["freeze_kind"] == "budget_stop"
    assert result.extra["stop_reason"] == "budget_stop"
    assert result.policy_source == POLICY


def test_invalid_policy_syntax_is_feedback_and_synthesis_continues():
    llm = MockLLMClient([
        _action("write_policy", source="def run(api):\n    broken = (\n"),
        _action("write_policy", source=POLICY),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=2,
                                 max_llm_calls=3, n_dev_cases=1)
    assert result.extra["frozen"]
    assert result.policy_source == POLICY
    second_prompt = llm.history[1]["prompt"]
    history = json.loads(second_prompt)["interaction_history"]
    assert "Syntax error" in history[0]["tool_result"]["error"]


def test_old_inspections_are_compact_and_complete_runs_remain_analyzable():
    """Long-trace history stays bounded while analysis and rereading retain every call."""
    source = 'def run(api):\n    print("diagnostic output")\n    for _ in range(200):\n        api.get_ee_pose()\n'
    llm = MockLLMClient([
        _action("write_policy", source=source),
        _action("run_experiment", case_indices=[0, 1, 2]),
        _action("inspect_run", run_id=0),
        _action("inspect_run", run_id=1),
        _action("inspect_run", run_id=99),
        _action("inspect_run", run_id=2),
        _action("run_analysis", script='def analyze(data):\n    return [[len(run["sdk_calls"]), run["stdout"]] for run in data]'),
        _action("inspect_run", run_id=0),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=3,
                                 max_llm_calls=9, n_dev_cases=3)
    after_third_inspection = json.loads(llm.history[6]["prompt"])["interaction_history"]
    inspected = [entry["tool_result"] for entry in after_third_inspection
                 if entry["action"]["tool"] == "inspect_run"]
    assert "sdk_calls" not in inspected[0]
    assert inspected[0]["n_sdk_calls"] == 200
    assert "run_analysis" in inspected[0]["details"]
    assert inspected[2] == {"error": "unknown run_id"}
    assert [run["run_id"] for run in inspected if "sdk_calls" in run] == [1, 2]
    assert result.extra["analysis_scripts"][0]["result"] == [[200, "diagnostic output\n"]] * 3
    final_history = json.loads(llm.history[-1]["prompt"])["interaction_history"]
    full_inspections = [entry["tool_result"] for entry in final_history
                        if entry["action"]["tool"] == "inspect_run"
                        and "sdk_calls" in entry["tool_result"]]
    assert [run["run_id"] for run in full_inspections] == [2, 0]
    assert all(len(run["sdk_calls"]) == 200 for run in result.extra["development_runs"])


def test_development_noise_seeds_match_case_indices_and_native_capabilities_are_declared():
    """Repeated cases share the evaluator's seed routing and report library capabilities."""
    backend = ToyRetuneTask()
    seeds = []
    make_episode = backend.make_episode

    def record_episode_seed(theta, seed):
        """Observe episode seeds while running the original toy backend."""
        seeds.append(seed)
        return make_episode(theta, seed=seed)

    backend.make_episode = record_episode_seed
    llm = MockLLMClient([
        _action("write_policy", source=POLICY),
        _action("run_experiment", case_indices=[2, 0, 2]),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(backend, llm, seed=7, max_rollouts=3,
                                 max_llm_calls=3, n_dev_cases=3)
    assert seeds == [9, 7, 9]
    api_doc = json.loads(llm.history[0]["prompt"])["robot_api"]
    assert "numpy" in api_doc and "scipy" in api_doc
    assert "NO `import`" not in api_doc
    adaptation = result.extra["platform_adaptation"]
    assert {"numpy", "scipy"}.issubset(adaptation["agentic_policy_imports"])
    assert adaptation["rebranch_policy_imports"] == []
    assert adaptation["rebranch_injected_modules"] == ["math"]
    assert adaptation["agentic_max_policy_lines"] == adaptation["rebranch_default_max_policy_lines"] == 50000


def test_missing_tool_response_is_feedback_and_later_inspection_continues():
    """Malformed responses are normalized before inspection history is compacted."""
    llm = MockLLMClient([
        _action("write_policy", source=POLICY),
        _action("run_experiment", case_indices=[0]),
        json.dumps({"run_id": 0}),
        _action("inspect_run", run_id=0),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=1,
                                 max_llm_calls=5, n_dev_cases=1)
    history = json.loads(llm.history[-1]["prompt"])["interaction_history"]
    assert history[2]["action"] == {"tool": "invalid", "arguments": {}}
    assert "response must be a JSON object with a string tool" in history[2]["tool_result"]["error"]
    assert history[3]["tool_result"]["sdk_calls"][0]["name"] == "zero_ee_wrench"
    assert result.extra["stop_reason"] == "explicit_freeze"


@pytest.mark.parametrize("invalid_request", [
    {"tool": "write_policy", "source": POLICY},
    {"tool": "inspect_task", "arguments": []},
])
def test_missing_or_nonobject_arguments_get_protocol_feedback(invalid_request):
    """Reject malformed envelopes, then accept the next canonical request unchanged."""
    llm = MockLLMClient([
        json.dumps(invalid_request),
        _action("write_policy", source=POLICY),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(ToyRetuneTask(), llm, max_rollouts=0,
                                 max_llm_calls=3, n_dev_cases=1)
    history = result.extra["tool_events"]
    assert "include an arguments JSON object" in history[0]["tool_result"]["error"]
    assert history[1]["tool_result"]["accepted"] is True
    assert result.policy_source == POLICY
    assert result.extra["stop_reason"] == "explicit_freeze"


def test_public_feedback_and_task_api_doc_reach_all_agentic_tools():
    """One evaluated result survives immediate, inspection, compact and analysis paths."""
    from rebranch.tasks import make_task

    feedback = {"protocol_version": "toy-terminal-v1", "hold_max_tilt_deg": 0.25,
                "all_footprints_in_2d_regions": False}
    backend = ToyRetuneTask()
    backend.extra_api_doc = make_task("t10_hidden_mass_balance_beam_a").extra_api_doc
    original_make = backend.make_episode
    evaluations = []

    def make_episode(theta, seed=0):
        """Attach public scoring feedback to the existing analytical toy evaluator."""
        episode = original_make(theta, seed=seed)
        original_evaluate = episode.evaluate

        def evaluate():
            """Record the single scoring call and return its public extension."""
            result = original_evaluate()
            result.public_feedback = feedback
            evaluations.append(seed)
            return result

        episode.evaluate = evaluate
        return episode

    backend.make_episode = make_episode
    llm = MockLLMClient([
        _action("inspect_task"),
        _action("write_policy", source=TEST),
        _action("run_experiment", case_indices=[0, 1, 2]),
        _action("inspect_run", run_id=0),
        _action("inspect_run", run_id=1),
        _action("inspect_run", run_id=2),
        _action("run_analysis", script='def analyze(data):\n    return [run["public_feedback"] for run in data]'),
        _action("freeze_policy"),
    ])
    result = run_agentic_genplan(backend, llm, seed=7, max_rollouts=3,
                                 max_llm_calls=8, n_dev_cases=3)
    assert evaluations == [7, 8, 9]
    events = result.extra["tool_events"]
    assert events[0]["tool_result"]["robot_api"].endswith(backend.extra_api_doc)
    assert "api.get_mass(name: str)" in backend.extra_api_doc
    assert [row["public_feedback"] for row in events[2]["tool_result"]["results"]] == [feedback] * 3
    assert "sdk_calls" not in events[3]["tool_result"]  # oldest full inspection compacted
    assert all(events[i]["tool_result"]["public_feedback"] == feedback for i in (3, 4, 5))
    assert result.extra["analysis_scripts"][0]["result"] == [feedback] * 3
    assert all(row["public_feedback"] == feedback and "theta" not in row
               for row in result.extra["development_runs"])
    for request in llm.history:
        prompt = json.loads(request["prompt"])
        assert prompt["robot_api"].endswith(backend.extra_api_doc)
        assert prompt["case_index_range"] == {"min_inclusive": 0, "max_inclusive": 2}
    assert result.dev_llm_calls == 8 and result.n_rollouts == 3
