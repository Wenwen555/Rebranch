from __future__ import annotations

from types import SimpleNamespace

from rebranch.core.branch import format_trace_feedback
from rebranch.eval.metrics import format_sdk_calls, summarize_trace_for_llm
from rebranch.exec.trace import CallRecord, Trace


def _trace(*, success=False, failure="insufficient_pull", error=None):
    obs = {
        "ee_x": 0.24667937678241736, "ee_y": 0.001, "ee_z": 0.095,
        "ee_qw": 1.0, "ee_qx": 0.0, "ee_qy": 0.0, "ee_qz": 0.0,
        "fx": 1.2, "fy": -0.4, "fz": 3.0,
        "tx": 0.1, "ty": 0.2, "tz": 0.3,
        "grip_width": 0.007, "grip_effort": 2.5, "grip_grasped": 1.0,
    }
    calls = [CallRecord(
        166, "get_object_pose", 89, 1, ("handle",), {"frame": "world"},
        {"pos": (0.36429440145537395, 0.00662377106418096, 0.095),
         "quat": [1.0, 0.0, 0.0, 0.0]}, obs,
    )]
    return Trace(theta={"SECRET_THETA_SENTINEL": 91}, seed=0, calls=calls,
                 success=success, score=0.5 if success else -0.2,
                 failure=None if success else failure, error=error)


def test_shared_sdk_summary_preserves_return_args_and_all_observation_fields():
    import rebranch.baselines.physcap_agent as online
    trace = _trace()
    ordinary = summarize_trace_for_llm(trace)
    branch = format_trace_feedback([trace])
    online_summary = summarize_trace_for_llm(trace, max_calls=8)
    for summary in (ordinary, branch, online_summary):
        assert "get_object_pose" in summary
        assert "'handle'" in summary
        assert "0.3643" in summary
        assert "0.2467" in summary
        assert "0.0066" in summary
        assert "grip_effort" in summary and "2.5000" in summary
        assert "grip_width" in summary and "0.0070" in summary
        assert "fx" in summary and "fy" in summary and "fz" in summary
        assert "tx" in summary and "ty" in summary and "tz" in summary
        assert "SECRET_THETA_SENTINEL" not in summary
    assert format_sdk_calls(trace).count("get_object_pose") == 1
    assert online.summarize_trace_for_llm is summarize_trace_for_llm


def test_branch_summary_keeps_success_trace_when_only_path_divergence_failed():
    trace = _trace(success=True)
    summary = format_trace_feedback([trace])
    assert "Trace counts: total=1, success=1, failed=0" in summary
    assert "get_object_pose" in summary
    assert "0.3643" in summary


def test_twin_second_prompt_pairs_draft_source_with_its_failure_trace(monkeypatch):
    import rebranch.baselines.twin_mpc as twin

    draft = "DRAFT_POLICY_MARKER = 1\ndef run(api):\n    pass\n"
    candidates = ["CANDIDATE_ONE_MARKER = 1\ndef run(api):\n    pass\n",
                  "CANDIDATE_TWO_MARKER = 1\ndef run(api):\n    pass\n"]
    prompts = []

    class FakeEvaluator:
        n_rollouts = 0
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def run_many(self, source, cases, seed):
            self.n_rollouts += len(cases)
            if source != draft:
                return [Trace(theta={}, seed=i, success=True) for i, _ in enumerate(cases)]
            return [Trace(theta={}, seed=i, success=False, score=-1,
                          failure=f"draft_failure_{i}",
                          calls=[CallRecord(i, "get_object_pose", 17, 1, ("handle",), {},
                                            {"pos": (0.3 + i / 100, 0.0, 0.1)},
                                            {"ee_x": 0.2, "grip_effort": 1.0})])
                    for i, _ in enumerate(cases)]

    class FakeLLM:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            code = candidates[len(prompts) - 1]
            return SimpleNamespace(text=f"```python\n{code}```", prompt_tokens=1,
                                   completion_tokens=1)

    backend = SimpleNamespace(
        param_space=SimpleNamespace(sample=lambda rng: {}), draft_policy=draft,
        task_description="mock task",
    )
    monkeypatch.setattr(twin, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(twin, "build_safe_probe_source", lambda backend: "PROBE")
    monkeypatch.setattr(twin, "probe_features", lambda trace, source: {})
    monkeypatch.setattr(twin, "tune_constants_cem", lambda *a, **k: (draft, 0, None))

    twin.run_twin_mpc(backend, FakeLLM(), max_rollouts=16, max_llm_calls=2,
                      n_twins=4)
    assert len(prompts) == 2
    assert "DRAFT_POLICY_MARKER = 1" in prompts[1]
    assert "CANDIDATE_ONE_MARKER" not in prompts[1]
    assert "draft_failure_1" in prompts[1]
    assert "get_object_pose" in prompts[1]
    assert "0.3100" in prompts[1]
    assert "SECRET_THETA_SENTINEL" not in prompts[1]
