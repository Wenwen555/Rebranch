from __future__ import annotations

from rebranch.baselines.oneshot_calib import OneShotCalibAgent
from rebranch.baselines.twin_mpc import TwinMPCAgent
from rebranch.baselines.physcap_agent import PhysCaPAgent
from rebranch.sim.toy_backend import ToyRetuneTask


def _policy(n: int) -> str:
    return f"def run(api):\n    for _ in range({n}):\n        api.get_ee_pose()\n"


def _counted_backend():
    backend = ToyRetuneTask()
    original = backend.make_episode
    episodes = []

    def make_episode(theta, seed=0):
        ep = original(theta, seed=seed)
        ep.protocol_version = "two-phase-toy-test"
        old_evaluate = ep.evaluate
        eval_count = [0]

        def evaluate():
            eval_count[0] += 1
            ep.api.sim_time += 0.125
            return old_evaluate()

        ep.evaluate = evaluate
        ep.test_eval_count = eval_count
        episodes.append(ep)
        return ep

    backend.make_episode = make_episode
    return backend, episodes


def _assert_one_final_evaluation(trace, episode):
    assert episode.test_eval_count[0] == 1
    assert trace.sim_time == episode.sim_time
    assert trace.protocol_version == "two-phase-toy-test"
    assert len(trace.calls) == 200
    assert trace.success is False
    assert trace.error == "max_calls (200) exceeded across episode phases"


def test_oneshot_runtime_uses_one_200_call_episode_budget(monkeypatch):
    import rebranch.baselines.oneshot_calib as module

    backend, episodes = _counted_backend()
    monkeypatch.setattr(module, "probe_features", lambda trace, source: {"x": 0.0})
    agent = OneShotCalibAgent(
        probe_source=_policy(75), twin_bank=[{}], probe_obs_bank=[{"x": 0.0}],
        calibrated_policies=[_policy(150)], calibrated_pred_ok=[False],
    )
    theta = backend.param_space.nominal()
    trace, _ = agent.evaluate_episode(backend, theta, seed=9)
    _assert_one_final_evaluation(trace, episodes[0])


def test_twin_runtime_uses_one_200_call_episode_budget(monkeypatch):
    import rebranch.baselines.twin_mpc as module

    backend, episodes = _counted_backend()
    monkeypatch.setattr(module, "probe_features", lambda trace, source: {"x": 0.0})
    theta = backend.param_space.nominal()
    agent = TwinMPCAgent(
        probe_source=_policy(75), twin_bank=[theta], probe_obs_bank=[{"x": 0.0}],
        candidate_policies=[_policy(150)], top_k_twins=1,
        runtime_sim_budget_per_episode=1,
    )
    trace, _ = agent.evaluate_episode(backend, theta, seed=10)
    # Candidate screening is a separate simulated episode; the real episode is last.
    assert len(episodes) == 2
    _assert_one_final_evaluation(trace, episodes[0])
    assert all(ep.test_eval_count[0] == 1 for ep in episodes)


def test_physcap_online_proxy_uses_one_200_call_episode_budget_without_llm():
    backend, episodes = _counted_backend()
    agent = PhysCaPAgent(
        llm=None, base_policy=_policy(150), probe_policy=_policy(75),
        runtime_llm_budget=0, max_requests_per_episode=0,
    )
    theta = backend.param_space.nominal()
    trace, metadata = agent.evaluate_episode(backend, theta, seed=11)
    _assert_one_final_evaluation(trace, episodes[0])
    assert metadata["runtime_llm_calls"] == 0


def test_run_policy_limits_remain_failures_after_successful_evaluation():
    from rebranch.exec.sandbox import run_policy
    from rebranch.sim.backend import EpisodeResult

    backend, episodes = _counted_backend()
    theta = backend.param_space.nominal()
    for limit_kwargs in ({"max_calls": 3}, {"max_lines": 3}):
        ep = backend.make_episode(theta, seed=12)
        eval_count = [0]

        def successful_final_evaluation():
            eval_count[0] += 1
            ep.api.sim_time += 0.125
            return EpisodeResult(True, 1.0, safety_violation="fake-marker")

        ep.evaluate = successful_final_evaluation
        trace = run_policy(_policy(10), ep, theta, seed=12, **limit_kwargs)
        assert eval_count[0] == 1
        assert trace.success is False
        assert trace.error is not None
        assert "max_" in trace.error
        assert trace.sim_time == ep.sim_time
        assert trace.safety_violation == "fake-marker"
