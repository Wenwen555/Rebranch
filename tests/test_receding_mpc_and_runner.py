from __future__ import annotations

from types import SimpleNamespace

import pytest

from rebranch.baselines.agentic_genplan import run_agentic_genplan
from rebranch.baselines.twin_mpc_receding import TwinMPCRecedingAgent
from rebranch.eval.runner import run_single_method
from rebranch.llm.client import MockLLMClient
from rebranch.sim.backend import SNAPSHOT_KEYS
from rebranch.sim.toy_backend import ToyRetuneTask


PROBE = """def run(api):
    api.get_ee_pose()
    # PROBE_FEATURES_V2
    api.get_gripper_state()
"""


class _ToyAPI:
    def __init__(self, episode):
        self.episode = episode

    def get_ee_pose(self):
        return {"pos": [self.episode.x, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}

    def get_gripper_state(self):
        return {"width": 0.05, "effort": 0.0, "grasped": False}

    def move_to(self, pos, **kwargs):
        self.episode.x = float(pos[0])
        self.episode.moves.append(self.episode.x)
        self.episode.sim_time += 0.1


class _ToyEpisode:
    protocol_version = "receding-toy-v1"

    def __init__(self):
        self.x = 0.0
        self.moves = []
        self.evaluations = 0
        self.sim_time = 0.0
        self.api = _ToyAPI(self)

    def snapshot(self):
        return {key: self.x if key == "ee_x" else 0.0 for key in SNAPSHOT_KEYS}

    def evaluate(self):
        self.evaluations += 1
        success = abs(self.x - 0.3) < 1e-9
        score = 1.0 if success else self.x - 1.0
        return SimpleNamespace(success=success, score=score,
                               failure=None if success else "not_done", safety_violation=None)


class _ToyTask:
    obs_scale = {key: 1.0 for key in SNAPSHOT_KEYS}

    def __init__(self):
        self.episodes = []
        self.param_space = ToyRetuneTask().param_space
        self.task_description = "Reach x=0.3 using public RobotAPI moves."
        self.draft_policy = ""

    def make_episode(self, theta, seed=0):
        ep = _ToyEpisode()
        self.episodes.append((seed, ep))
        return ep


A = """def run(api):
    pose = api.get_ee_pose()["pos"]
    if pose[0] < 0.15:
        api.move_to([0.1, 0.0, 0.0])
        api.move_to([0.1, 0.0, 0.0])
    else:
        api.move_to([0.3, 0.0, 0.0])
        api.move_to([0.3, 0.0, 0.0])
"""
B = """def run(api):
    pose = api.get_ee_pose()["pos"]
    if pose[0] < 0.15:
        api.move_to([0.2, 0.0, 0.0])
        api.move_to([0.2, 0.0, 0.0])
    else:
        api.move_to([0.3, 0.0, 0.0])
        api.move_to([0.3, 0.0, 0.0])
"""


def _agent(task, *, max_calls=20, budget=10, candidates=(A, B)):
    return TwinMPCRecedingAgent(PROBE, [{"sample": 0.0}],
        [{key: 0.0 for key in SNAPSHOT_KEYS}], list(candidates),
        prediction_horizon=2, top_k_twins=1,
        runtime_sim_budget_per_episode=budget, max_calls=max_calls)


def test_current_observation_changes_candidate_choice_after_public_prefix_replay():
    backend = _ToyTask()
    trace, metadata = _agent(backend).evaluate_episode(backend, {"hidden": 0.9}, seed=7)
    assert [p["candidate_index"] for p in metadata["planning_steps"][:2]] == [1, 0]
    assert backend.episodes[0][1].moves[:2] == [0.2, 0.3]
    assert trace.success and backend.episodes[0][1].evaluations == 1
    assert metadata["runtime_replay_api_calls"] > 0
    # Forecasts use one fixed seed per particle, paired across decisions/candidates.
    forecast_seeds = [seed for seed, _ in backend.episodes if seed >= 50000]
    assert len(set(forecast_seeds)) == 1


def test_actual_api_budget_exhaustion_fails_and_evaluates_once():
    backend = _ToyTask()
    short_probe = "def run(api):\n    api.get_ee_pose()\n    # PROBE_FEATURES_V2\n    api.get_gripper_state()\n"
    agent = TwinMPCRecedingAgent(short_probe, [{"sample": 0.0}],
        [{key: 0.0 for key in SNAPSHOT_KEYS}], [A], prediction_horizon=2,
        top_k_twins=1, runtime_sim_budget_per_episode=5, max_calls=2)
    trace, metadata = agent.evaluate_episode(backend, {"hidden": 0.9}, seed=8)
    assert metadata["stop_reason"] == "actual_api_budget_exhausted"
    assert trace.success is False and trace.error == "actual_api_budget_exhausted"
    assert backend.episodes[0][1].evaluations == 1


def test_normal_candidate_completion_is_not_an_automatic_error():
    backend = _ToyTask()
    no_action = "def run(api):\n    api.get_ee_pose()\n"
    # This environment starts complete; a candidate that returns without an
    # actuator call is a normal completion and should be evaluated as such.
    class AlreadyDoneTask(_ToyTask):
        def make_episode(self, theta, seed=0):
            ep = super().make_episode(theta, seed)
            ep.x = 0.3
            return ep
    backend = AlreadyDoneTask()
    trace, metadata = TwinMPCRecedingAgent(PROBE, [{"sample": 0.0}],
        [{key: 0.0 for key in SNAPSHOT_KEYS}], [no_action], prediction_horizon=2,
        top_k_twins=1, runtime_sim_budget_per_episode=2, max_calls=20).evaluate_episode(backend, {}, 1)
    assert metadata["stop_reason"] == "candidate_completed"
    assert trace.success and trace.error is None
    assert backend.episodes[0][1].evaluations == 1


def test_runner_exposes_agentic_method_and_frozen_runtime_without_llm():
    actions = [
        '{"tool":"inspect_task","arguments":{}}',
        '{"tool":"write_policy","arguments":{"source":"def run(api):\\n    api.get_ee_pose()\\n"}}',
        '{"tool":"freeze_policy","arguments":{}}',
    ]
    out = run_single_method("toy_retune", "agentic_genplan", MockLLMClient(actions),
        max_rollouts=1, max_llm_calls=3, n_hidden=0, n_ood=0,
        output_dir="/tmp/rebranch-agentic-runner-test")
    assert out["method"] == "agentic_genplan"
    assert out["result"].extra["identity"] == "platform_adaptation"
    assert out["result"].extra["frozen"]
    assert out["result"].dev_llm_calls == 3


def test_runner_registers_receding_mpc_and_reports_horizon(tmp_path):
    out = run_single_method("toy_retune", "twin_mpc_receding", MockLLMClient([]),
        max_rollouts=8, max_llm_calls=0, n_hidden=0, n_ood=0,
        twin_mpc_prediction_horizon=2, output_dir=tmp_path)
    assert out["method"] == "twin_mpc_receding"
    assert out["result"].extra["identity"] == "custom_control"
    assert out["result"].extra["prediction_horizon_actions"] == 2
    assert out["result"].extra["first_action_only"] is True


def test_horizon_one_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        TwinMPCRecedingAgent(PROBE, [], [], [A], prediction_horizon=1)
