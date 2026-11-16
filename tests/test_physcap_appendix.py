from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rebranch.baselines.physcap_appendix import PhysCaPAppendixAgent, PhysCaPTaskBackend, _candidate_key
from rebranch.exec.sandbox import run_policy
from rebranch.llm.client import CodexCLIClient, DiskCacheLLM, LimitedLLMClient, LLMResponse, LLMBudgetExceededError, MockLLMClient
from rebranch.sim.backend import EpisodeResult, SNAPSHOT_KEYS
from rebranch.sim.render import render_mujoco_rgb_png


class _API:
    def __init__(self, episode):
        self.episode = episode
        self.pos = [0.30, 0.0, 0.20]
        self.quat = [0.0, 1.0, 0.0, 0.0]
        self.actions = []

    def get_ee_pose(self):
        return {"pos": list(self.pos), "quat": list(self.quat)}

    def get_object_pose(self, name):
        assert name in {"can-a", "can-b"}
        return {"pos": [0.34, 0.02, 0.08], "quat": [1.0, 0.0, 0.0, 0.0]}

    def open_gripper(self, width=0.07):
        self.actions.append(("open", width))
        self.episode.sim_time += 0.05

    def close_gripper(self, force=8.0):
        self.actions.append(("close", force))
        self.episode.sim_time += 0.05

    def move_to(self, pos, quat=None, speed=0.08):
        self.actions.append(("move_to", tuple(pos)))
        self.pos = list(pos)
        if quat is not None:
            self.quat = list(quat)
        self.episode.sim_time += 0.1

    def move_compliant(self, pos, quat=None, free_axes="", speed=0.04, duration=None):
        self.actions.append(("hold", tuple(pos), duration))
        self.pos = list(pos)
        self.episode.sim_time += float(duration or 0.0)


class _Episode:
    protocol_version = "test"

    def __init__(self):
        self.sim_time = 0.0
        self.evaluations = 0
        self.api = _API(self)
        self.enabled = False

    def enable_physcap_mass_reveal(self):
        self.enabled = True

    def get_mass(self, name):
        assert self.enabled
        self.api.actions.append(("read_mass", name))
        self.sim_time += 0.6
        return 0.237

    def render_rgb(self, **kwargs):
        return b"fake png image payload"

    def snapshot(self):
        return {key: 0.0 for key in SNAPSHOT_KEYS}

    def evaluate(self):
        self.evaluations += 1
        return EpisodeResult(success=True, score=1.0)


class _Backend:
    name = "appendix_test"
    task_description = "Measure the mass of public object IDs can-a and can-b."
    requires_branching = False
    param_space = None
    obs_scale = {}
    obs_api_map = {}
    draft_policy = ""

    def __init__(self):
        self.episode = None

    def make_episode(self, theta, seed=0):
        self.episode = _Episode()
        return self.episode

    def hidden_test_set(self, seed=2026, n=50):
        return []

    def ood_test_set(self, seed=3026, n=10):
        return []

    def oracle_policy(self, theta):
        return "def run(api):\\n    pass\\n"

    def conservative_policy(self):
        return "def run(api):\\n    pass\\n"

    def latent_mode(self, theta):
        return "nominal"


def test_complete_visual_measurement_cycle_appends_knowledge_and_preserves_selected_candidate():
    insufficient = {
        "sufficient": False, "task_complete": False,
        "exploration_candidates": [
            {"name": "can-a", "action": "measure_mass", "description": "mass A", "expected_info": "mass", "parameters": {"x": 1}, "estimated_cost": "low"},
            {"name": "can-b", "action": "measure_mass", "description": "mass B", "expected_info": "mass", "parameters": {"x": 2}, "estimated_cost": "medium"},
        ],
    }
    sufficient = {"sufficient": True, "task_complete": False, "exploration_candidates": []}
    responses = [
        "Initial scene knowledge.",
        json.dumps(insufficient),
        json.dumps({"scored_candidates": [
            {"name": "can-a", "action": "measure_mass", "reason": "less", "priority_score": 2},
            {"name": "can-b", "action": "measure_mass", "reason": "more", "priority_score": 9},
        ]}),
        "```python\ndef run(api):\n    api.get_mass(name='can-a')\n```",
        "VDM_SENTINEL",
        json.dumps(insufficient),
        json.dumps({"scored_candidates": [
            {"name": "can-b", "action": "measure_mass", "reason": "more", "priority_score": 9},
        ]}),
        "```python\ndef run(api):\n    api.get_mass('can-b')\n```",
        "VDM_SECOND",
        json.dumps(sufficient),
        "```python\ndef run(api):\n    print('FINISH')\n```",
    ]
    llm = MockLLMClient(responses)
    backend = PhysCaPTaskBackend(_Backend())
    trace, meta = PhysCaPAppendixAgent(llm, max_turns=5).evaluate_episode(backend, {}, seed=7)

    assert trace.success and trace.error is None
    assert backend._inner.episode.evaluations == 1
    assert trace.stdout.endswith("FINISH\n")
    assert [call.index for call in trace.calls] == list(range(len(trace.calls)))
    assert trace.calls[0].name == "get_mass"
    # First call measured can-a while can-b was selected; it remains pending.
    assert meta["unrecorded_attempts"]["('can b', 'measure mass')"] == 1
    assert meta["measurements"][0]["name"] == "can-a"
    assert meta["measurements"][1]["name"] == "can-b"
    assert meta["measurements"][1]["value"] == pytest.approx(0.237)
    assert meta["measurements"][1]["sim_time_s"] >= 1.5  # approach, lift/hold, return, release, retreat
    actions = backend._inner.episode.api.actions
    rounded = [
        (row[0], tuple(round(v, 3) for v in row[1])) if row[0] == "move_to"
        else (row[0], tuple(round(v, 3) for v in row[1]), row[2]) if row[0] == "hold"
        else row
        for row in actions
    ]
    assert rounded == [
        ("open", 0.07), ("move_to", (0.34, 0.02, 0.18)),
        ("move_to", (0.34, 0.02, 0.08)), ("close", 8.0),
        ("move_to", (0.34, 0.02, 0.23)),
        ("hold", (0.34, 0.02, 0.23), 0.6), ("read_mass", "can-a"),
        ("move_to", (0.34, 0.02, 0.085)),
        ("hold", (0.34, 0.02, 0.085), 0.3), ("open", 0.07),
        ("move_to", (0.34, 0.02, 0.23)), ("move_to", (0.30, 0.0, 0.20)),
        ("open", 0.07), ("move_to", (0.34, 0.02, 0.18)),
        ("move_to", (0.34, 0.02, 0.08)), ("close", 8.0),
        ("move_to", (0.34, 0.02, 0.23)),
        ("hold", (0.34, 0.02, 0.23), 0.6), ("read_mass", "can-b"),
        ("move_to", (0.34, 0.02, 0.085)),
        ("hold", (0.34, 0.02, 0.085), 0.3), ("open", 0.07),
        ("move_to", (0.34, 0.02, 0.23)), ("move_to", (0.30, 0.0, 0.20)),
    ]

    roles = [event["role"] for event in meta["llm_events"]]
    assert roles == ["scene_knowledge", "planner", "prioritizer", "coding_single_step",
                     "visual_differencing", "planner", "prioritizer", "coding_single_step",
                     "visual_differencing", "planner", "coding_full_task"]
    assert len(meta["llm_events"][4]["images_sha256"]) == 2
    assert "VDM_SENTINEL" in meta["llm_events"][7]["prompt"]
    assert "name (the object to act on): can-b" in meta["llm_events"][7]["prompt"]
    assert "can-a" in meta["llm_events"][5]["prompt"]
    assert "0.237" in meta["llm_events"][5]["prompt"]
    assert "get_mass(name)" in meta["llm_events"][7]["prompt"]
    assert "can-a" in meta["llm_events"][1]["prompt"]
    second_rank_prompt = [event for event in meta["llm_events"] if event["role"] == "prioritizer"][1]["prompt"]
    candidate_text = second_rank_prompt.split("\n\nCandidates:\n", 1)[1].split("\n\nScene knowledge:", 1)[0]
    assert [candidate["name"] for candidate in json.loads(candidate_text)] == ["can-b"]


def test_capture_output_exposes_print_only_when_requested():
    episode = _Episode()
    source = "def run(api):\n    print('reading', 3)\n"
    trace = run_policy(source, episode, {}, capture_output=True, evaluate=False)
    assert trace.stdout == "reading 3\n"
    legacy = run_policy(source, episode, {}, capture_output=False, evaluate=False)
    assert legacy.error and "NameError" in legacy.error
    assert legacy.stdout == ""


def test_old_text_cache_key_and_image_specific_keys():
    cache = DiskCacheLLM(MockLLMClient(["ok"]), cache_dir=Path("/tmp/physcap-cache-test"))
    payload = {
        "inner": "MockLLMClient", "model": "", "effort": "", "system": "s",
        "prompt": "p", "temperature": 0.2,
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert cache._cache_key("p", "s", 0.2) == expected
    assert cache._cache_key("p", "s", 0.2, (b"image-a",)) != cache._cache_key("p", "s", 0.2, (b"image-b",))


def test_codex_cli_places_multiple_images_before_separator_and_stdin(monkeypatch):
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = list(cmd)
        observed["input"] = kwargs["input"]
        image_flag = cmd.index("-i")
        paths = cmd[image_flag + 1:cmd.index("--")]
        observed["images"] = [Path(path).read_bytes() for path in paths]
        observed["work_dir"] = cmd[cmd.index("--cd") + 1]
        observed["image_paths"] = paths
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}), stderr="")

    monkeypatch.setattr("rebranch.llm.client.subprocess.run", fake_run)
    client = CodexCLIClient(codex_bin="/bin/true")
    response = client.complete("prompt", images=(b"one", b"two"))
    cmd = observed["cmd"]
    flag = cmd.index("-i")
    assert cmd[flag + 1:flag + 3] == [str(Path(cmd[flag + 1])), str(Path(cmd[flag + 2]))]
    assert cmd[flag + 3:flag + 5] == ["--", "-"]
    assert observed["images"] == [b"one", b"two"]
    assert observed["image_paths"] == [str(Path(observed["work_dir"]) / "image_0.png"),
                                       str(Path(observed["work_dir"]) / "image_1.png")]
    assert observed["input"] == "prompt"
    assert response.text == "ok"


def test_candidate_keys_follow_official_normalization_and_description_fallback():
    assert _candidate_key({"name": "Can_A", "action": "Measure-Mass"}) == (
        "can a", "measure mass"
    )
    assert _candidate_key({"name": "CAN-A", "description": "Check_Mass", "action": "other"}) == (
        "can a", "check mass"
    )


def test_prioritizer_schema_retries_three_invalid_outputs_and_ignores_early_task_complete():
    candidates = [{"name": "can-a", "action": "measure_mass", "description": "Measure mass"}]
    first_plan = {"task_complete": True, "sufficient": False, "exploration_candidates": candidates}
    next_plan = {"task_complete": False, "sufficient": True, "exploration_candidates": []}
    llm = MockLLMClient([
        "initial knowledge", json.dumps(first_plan), "not json", "{}", json.dumps({"scored_candidates": []}),
        "```python\ndef run(api):\n    pass\n```", "VDM",
        json.dumps(next_plan), "```python\ndef run(api):\n    print('FINISH')\n```",
    ])
    trace, meta = PhysCaPAppendixAgent(llm, max_turns=3).evaluate_episode(
        PhysCaPTaskBackend(_Backend()), {}, seed=4
    )
    assert trace.success
    assert meta["runtime_llm_calls"] == 9
    assert len(meta["prioritizer_fallbacks"]) == 1
    assert meta["prioritizer_fallbacks"][0]["attempts"] == 3
    assert len(meta["prioritizer_fallbacks"][0]["invalid_attempts"]) == 3
    rank_events = [event for event in meta["llm_events"] if event["role"] == "prioritizer"]
    assert len(rank_events) == 3
    assert all("score mode" in event["prompt"] for event in rank_events)
    assert all('"scored_candidates"' in event["prompt"] for event in rank_events)
    assert "0=lowest" in rank_events[0]["prompt"] and "10=highest" in rank_events[0]["prompt"]
    assert "can-a" in rank_events[0]["prompt"]
    assert meta["llm_events"][1]["role"] == "planner"
    assert "exact public object IDs" in meta["llm_events"][1]["prompt"]


def test_direct_mujoco_render_does_not_step_or_change_state():
    import mujoco

    xml = """<mujoco><worldbody><geom name='floor' type='plane' size='1 1 0.1'/><body name='box' pos='0 0 0.2'><freejoint/><geom type='box' size='0.05 0.05 0.05' mass='0.2'/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = (float(data.time), data.qpos.copy(), data.qvel.copy())
    image = render_mujoco_rgb_png(model, data, width=640, height=480)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert float(data.time) == before[0]
    np.testing.assert_array_equal(data.qpos, before[1])
    np.testing.assert_array_equal(data.qvel, before[2])


def test_runner_registers_appendix_method_without_development_calls(monkeypatch, tmp_path):
    import rebranch.eval.runner as runner

    backend = _Backend()
    monkeypatch.setattr(runner, "load_task_backend", lambda _name: backend)
    llm = MockLLMClient(["unused"])
    output = runner.run_single_method(
        task_name="appendix_test", method="physcap_appendix", llm=llm,
        n_hidden=0, n_ood=0, workers=1, output_dir=tmp_path,
    )
    assert output["method"] == "physcap_appendix"
    assert output["result"].method == "physcap_appendix"
    assert output["result"].n_rollouts == 0
    assert output["result"].dev_llm_calls == 0
    assert llm.stats["calls"] == 0


def test_static_error_candidate_is_feedback_and_later_valid_final_code_recovers():
    plan = {"sufficient": False, "task_complete": False, "exploration_candidates": [
        {"name": "can-a", "action": "measure_mass", "description": "mass", "expected_info": "mass", "parameters": {}, "estimated_cost": "low"}
    ]}
    ranked = {"scored_candidates": [{"name": "can-a", "action": "measure_mass", "reason": "needed", "priority_score": 8}]}
    llm = MockLLMClient([
        "Initial scene.", json.dumps(plan), json.dumps(ranked),
            "```python\nimport os\ndef run(api):\n    pass\n```",
        "No visible change.",
        json.dumps({"sufficient": True, "task_complete": False}),
        "```python\ndef run(api):\n    print('FINISH')\n```",
    ])
    backend = PhysCaPTaskBackend(_Backend())
    trace, meta = PhysCaPAppendixAgent(llm).evaluate_episode(backend, {}, seed=2)
    assert trace.success and trace.error is None
    assert backend._inner.episode.evaluations == 1
    assert meta["failed_candidates"] == [("can a", "measure mass")]
    assert "StaticCheckError" in meta["llm_events"][6]["prompt"]


def test_static_error_in_final_task_is_episode_failure():
    llm = MockLLMClient([
        "Initial scene.", json.dumps({"sufficient": True, "task_complete": False}),
        "```python\nimport os\ndef run(api):\n    pass\n```",
    ])
    backend = PhysCaPTaskBackend(_Backend())
    trace, _meta = PhysCaPAppendixAgent(llm).evaluate_episode(backend, {}, seed=3)
    assert backend._inner.episode.evaluations == 1
    assert not trace.success
    assert trace.error and "StaticCheckError" in trace.error


def test_shared_runtime_llm_budget_is_reserved_atomically():
    inner = MockLLMClient(["ok"])
    limited = LimitedLLMClient(inner, limit=3)

    def call(_index):
        try:
            limited.complete("prompt")
            return "completed"
        except LLMBudgetExceededError:
            return "budget_exhausted"

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(call, range(24)))
    assert results.count("completed") == 3
    assert results.count("budget_exhausted") == 21
    assert limited.attempts == 3
    assert inner.stats["calls"] == 3
