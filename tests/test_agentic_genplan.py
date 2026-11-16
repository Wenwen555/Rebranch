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
    return json.dumps({"tool": tool, **kwargs})


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
