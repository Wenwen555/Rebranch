"""Unit and integration tests for ReBranch foundation modules."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from rebranch.api import (
    API_DOC,
    PHYSCAP_EXTRA_PRIMITIVES,
    PRIMITIVES,
    EpisodeTerminated,
    JointNoiseModel,
    ProprioceptiveEstimator,
    RobotAPI,
)
from rebranch.exec import (
    BudgetExceededError,
    Evaluator,
    StaticCheckError,
    check_policy_source,
    run_policy,
)
from rebranch.llm import (
    CodexCLIClient,
    DiskCacheLLM,
    MockLLMClient,
    build_branch_prompt,
    build_physcap_prompt,
    build_repair_prompt,
    build_single_twin_prompt,
    build_uniform_dr_prompt,
    extract_python_code,
)
from rebranch.sim import (
    SNAPSHOT_KEYS,
    Continuous,
    Discrete,
    EpisodeResult,
    ParamSpace,
    ToyBranchTask,
    ToyRetuneTask,
)


def _six_dof_arm_xml(payload_mass: float = 0.0) -> str:
    payload_geom = (
        f'<geom name="payload" type="sphere" size="0.02" pos="0 0 0" mass="{payload_mass}"/>'
        if payload_mass > 0.0
        else ""
    )
    return f"""
<mujoco model="test_6dof">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint armature="0.05" damping="0.2"/>
  </default>
  <worldbody>
    <body name="l1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02" mass="0.3"/>
      <body name="l2" pos="0 0 0.1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.4"/>
        <body name="l3" pos="0.25 0 0">
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.20 0 0" size="0.02" mass="0.3"/>
          <body name="l4" pos="0.20 0 0">
            <joint name="j4" type="hinge" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015" mass="0.15"/>
            <body name="l5" pos="0.05 0 0">
              <joint name="j5" type="hinge" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015" mass="0.15"/>
              <body name="l6" pos="0.05 0 0">
                <joint name="j6" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.015" pos="0 0 0" mass="0.1"/>
                <site name="tcp" pos="0.03 0 0" euler="0 1.5707963 0"/>
                {payload_geom}
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="200" kv="20"/>
    <position name="a2" joint="j2" kp="200" kv="20"/>
    <position name="a3" joint="j3" kp="200" kv="20"/>
    <position name="a4" joint="j4" kp="100" kv="10"/>
    <position name="a5" joint="j5" kp="100" kv="10"/>
    <position name="a6" joint="j6" kp="100" kv="10"/>
  </actuator>
</mujoco>
"""


def test_primitives_and_api_doc() -> None:
    assert len(PRIMITIVES) == 10
    assert PHYSCAP_EXTRA_PRIMITIVES == ("get_mass", "get_stiffness")
    for p in PRIMITIVES:
        assert p in API_DOC
        assert hasattr(RobotAPI, p)
    assert "zero_ee_wrench" in API_DOC
    assert "[0.22, 0.48]" in API_DOC
    assert "[-0.22, 0.22]" in API_DOC
    assert "[1.0, 20.0]" in API_DOC
    assert "[0.0, 0.07]" in API_DOC
    err = EpisodeTerminated("crushed")
    assert err.reason == "crushed"
    assert str(err) == "crushed"


def test_sim_init_does_not_eagerly_import_mujoco_modules() -> None:
    assert "rebranch.sim.control" not in sys.modules
    assert "rebranch.sim.mujoco_episode" not in sys.modules


def test_proprioception_fk_jacobian_and_300g_sign(tmp_path: Path) -> None:
    est_xml = tmp_path / "arm_est.xml"
    est_xml.write_text(_six_dof_arm_xml(payload_mass=0.0))
    sim_xml = tmp_path / "arm_sim.xml"
    sim_xml.write_text(_six_dof_arm_xml(payload_mass=0.30))

    est = ProprioceptiveEstimator(model_path=est_xml, tcp_site="tcp", damping=0.01)

    q_test = np.array([0.2, 0.3, -0.5, 0.4, 0.2, -0.3])
    pos, quat = est.fk(q_test)
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    assert math.isclose(np.linalg.norm(quat), 1.0, rel_tol=1e-5)

    J = est.jacobian(q_test)
    assert J.shape == (6, 6)
    eps = 1e-6
    for j in range(6):
        qp = q_test.copy()
        qm = q_test.copy()
        qp[j] += eps
        qm[j] -= eps
        p_plus, _ = est.fk(qp)
        p_minus, _ = est.fk(qm)
        assert np.allclose(J[:3, j], (p_plus - p_minus) / (2 * eps), atol=1e-4)

    # Simulate unloaded tare then 300 g hanging payload in MuJoCo.
    m_est = mujoco.MjModel.from_xml_path(str(est_xml))
    d_est = mujoco.MjData(m_est)
    q_ref = np.array([0.2, -0.5, 0.8, 0.3, 0.6, 0.2])
    d_est.ctrl[:] = q_ref
    d_est.qpos[:] = q_ref
    for _ in range(1500):
        mujoco.mj_step(m_est, d_est)
    est.tare(d_est.qpos[:6], d_est.qfrc_actuator[:6])

    m_sim = mujoco.MjModel.from_xml_path(str(sim_xml))
    d_sim = mujoco.MjData(m_sim)
    d_sim.ctrl[:] = q_ref
    d_sim.qpos[:] = q_ref
    for _ in range(1500):
        mujoco.mj_step(m_sim, d_sim)

    w_world = est.estimate(
        d_sim.qpos[:6], d_sim.qvel[:6], d_sim.qfrc_actuator[:6], frame="world"
    )
    assert set(w_world.keys()) == {"force", "torque", "joint_delta"}
    assert len(w_world["joint_delta"]) == 6
    # Critical sign: 300 g hanging mass gives fz approx -2.94 N
    assert w_world["force"][2] == pytest.approx(-2.943, rel=0.10)

    w_ee = est.estimate(
        d_sim.qpos[:6], d_sim.qvel[:6], d_sim.qfrc_actuator[:6], frame="ee"
    )
    assert np.linalg.norm(w_ee["force"]) == pytest.approx(
        np.linalg.norm(w_world["force"]), rel=1e-4
    )


def test_joint_noise_model() -> None:
    noise = JointNoiseModel(enabled=True)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    noise.reset(rng1)

    q = np.zeros(6)
    qdot = np.array([0.1, -0.2, 0.0, 0.05, -0.05, 0.3])
    tau = np.array([1.0, -2.0, 1.5, 0.3, -0.4, 0.2])
    out1 = noise.apply_arm(q, qdot, tau, rng1)
    assert out1.shape == (6,)
    assert not np.allclose(out1, tau)

    noise2 = JointNoiseModel(enabled=True)
    noise2.reset(rng2)
    out2 = noise2.apply_arm(q, qdot, tau, rng2)
    assert np.allclose(out1, out2)

    w, eff = noise.apply_gripper(0.05, 8.0, rng1)
    assert 0.0 <= w <= 0.07
    assert isinstance(eff, float)

    clean = JointNoiseModel(enabled=False)
    clean.reset(rng1)
    assert np.allclose(clean.apply_arm(q, qdot, tau, rng1), tau)
    assert clean.apply_gripper(0.05, 8.0, rng1) == (0.05, 8.0)


def test_param_space_and_episode_result() -> None:
    space = ParamSpace(
        [
            Continuous("x", 0.0, 10.0, 5.0),
            Discrete("mode", ("a", "b"), "a"),
            Continuous("nuisance", -1.0, 1.0, 0.0),
        ]
    )
    assert space.nominal() == {"x": 5.0, "mode": "a", "nuisance": 0.0}
    rng = np.random.default_rng(7)
    s1 = space.sample(rng)
    assert 0.0 <= s1["x"] <= 10.0
    assert s1["mode"] in ("a", "b")

    sub = space.with_active({"x"})
    s2 = sub.sample(rng)
    assert s2["mode"] == "a"
    assert s2["nuisance"] == 0.0
    assert sub.distance({"x": 0.0, "mode": "a", "nuisance": 0.0}, {"x": 10.0, "mode": "b", "nuisance": 1.0}) == pytest.approx(1.0)
    mid = space.interpolate({"x": 0.0, "mode": "a", "nuisance": 0.0}, {"x": 10.0, "mode": "b", "nuisance": 1.0}, 0.25)
    assert mid == {"x": 2.5, "mode": "a", "nuisance": 0.25}

    with pytest.raises(ValueError):
        EpisodeResult(success=True, score=0.0)
    with pytest.raises(ValueError):
        EpisodeResult(success=False, score=0.2)
    with pytest.raises(ValueError):
        EpisodeResult(success=True, score=1.5)


def test_toy_backends_and_anti_cheat() -> None:
    assert "slip_distance" not in SNAPSHOT_KEYS
    for task in (ToyRetuneTask(), ToyBranchTask()):
        nominal = task.param_space.nominal()
        nom_trace = run_policy(task.draft_policy, task.make_episode(nominal, seed=0), nominal, seed=0)
        assert nom_trace.success

        hidden = task.hidden_test_set(seed=2026, n=50)
        ood = task.ood_test_set(seed=3026, n=10)
        assert len(hidden) == 50
        assert len(ood) == 10

        oracle_ok = sum(
            run_policy(task.oracle_policy(th), task.make_episode(th, seed=i), th, seed=i).success
            for i, th in enumerate(hidden)
        )
        assert oracle_ok / len(hidden) >= 0.90

        cons_ok = sum(
            run_policy(task.conservative_policy(), task.make_episode(th, seed=i), th, seed=i).success
            for i, th in enumerate(hidden)
        )
        assert cons_ok / len(hidden) < 0.50

        # Snapshot strict keys check
        for call in nom_trace.calls:
            assert tuple(call.obs.keys()) == SNAPSHOT_KEYS

    # ToyBranchTask proprioceptive separability (fragile vs heavy) via SDK only
    branch_task = ToyBranchTask()
    probe_policy = """
def run(api):
    api.zero_ee_wrench()
    p = api.get_object_pose("cup")["pos"]
    api.open_gripper(0.065)
    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)
    api.move_to([p[0], p[1], p[2]], speed=0.04)
    api.close_gripper(force=3.0)
    st = api.get_gripper_state()
    if st["width"] < 0.051:
        api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)
    else:
        api.close_gripper(force=16.0)
        api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)
"""
    hidden_b = branch_task.hidden_test_set(seed=2026, n=50)
    branch_ok = sum(
        run_policy(probe_policy, branch_task.make_episode(th, seed=i), th, seed=i).success
        for i, th in enumerate(hidden_b)
    )
    assert branch_ok == 50


def test_sandbox_ast_rejections_and_determinism() -> None:
    bad_sources = [
        "import math\ndef run(api):\n    pass\n",
        "from os import path\ndef run(api):\n    pass\n",
        "def run(api):\n    api._raw_api\n",
        "def run(api):\n    eval('1+1')\n",
        "def run(api):\n    exec('x=1')\n",
        "def run(api):\n    open('/etc/passwd')\n",
        "def run(api):\n    getattr(api, 'move_to')\n",
        "def run(api):\n    globals()\n",
        "def run(api):\n    locals()\n",
        "def run(api):\n    __import__('os')\n",
        "def run(api):\n    compile('1', 'x', 'eval')\n",
        "def helper():\n    pass\n",
        "def run(api)\n    pass\n",
    ]
    for src in bad_sources:
        with pytest.raises(StaticCheckError):
            check_policy_source(src)

    task = ToyRetuneTask()
    src_with_math = """
def run(api):
    x = math.cos(0.0) * 0.35
    api.zero_ee_wrench()
    for _ in range(2):
        api.get_ee_pose()
    api.move_to([x, 0.0, 0.04], speed=0.08)
    api.close_gripper(force=10.0)
    api.move_to([x, 0.0, 0.16], speed=0.02)
"""
    theta = task.param_space.nominal()
    t1 = run_policy(src_with_math, task.make_episode(theta, seed=123), theta, seed=123)
    t2 = run_policy(src_with_math, task.make_episode(theta, seed=123), theta, seed=123)
    assert t1.success == t2.success
    assert t1.score == t2.score
    assert t1.lines == t2.lines
    assert t1.path == t2.path
    assert [c.name for c in t1.calls] == [c.name for c in t2.calls]
    assert [c.occurrence for c in t1.calls] == [c.occurrence for c in t2.calls]
    assert [c.obs for c in t1.calls] == [c.obs for c in t2.calls]
    assert t1.calls[1].occurrence == 1
    assert t1.calls[2].occurrence == 2

    # Max calls / max lines budget
    loop_src = "def run(api):\n    while True:\n        api.get_ee_pose()\n"
    t_calls = run_policy(loop_src, task.make_episode(theta, seed=0), theta, max_calls=5)
    assert not t_calls.success
    assert "max_calls" in (t_calls.error or "")

    spin_src = "def run(api):\n    x = 0\n    while True:\n        x += 1\n"
    t_lines = run_policy(spin_src, task.make_episode(theta, seed=0), theta, max_lines=50)
    assert not t_lines.success
    assert "max_lines" in (t_lines.error or "")


def test_evaluator_multiprocessing_and_budget() -> None:
    task = ToyRetuneTask()
    cases = task.hidden_test_set(seed=99, n=8)
    with Evaluator(task, workers=1, budget_limit=20) as ev_seq:
        traces_seq = ev_seq.run_many(task.conservative_policy(), cases, seed=5)
        assert ev_seq.n_rollouts == 8

    with Evaluator(task, workers=2, budget_limit=10) as ev_mp:
        traces_mp = ev_mp.run_many(task.conservative_policy(), cases, seed=5)
        assert ev_mp.n_rollouts == 8
        assert [(t.success, t.score, t.failure) for t in traces_mp] == [
            (t.success, t.score, t.failure) for t in traces_seq
        ]
        with pytest.raises(BudgetExceededError):
            ev_mp.run_many(task.draft_policy, cases[:3], seed=0)


def test_prompts_no_theta_leak_and_llm_wrappers(tmp_path: Path) -> None:
    task = ToyBranchTask()
    p1 = build_repair_prompt(task.task_description, task.draft_policy, "slipped at z=0.05")
    p2 = build_branch_prompt(task.task_description, task.draft_policy, "2 cases conflict", "grip_width differs")
    p3 = build_single_twin_prompt(task.task_description, task.draft_policy, "slipped")
    p4 = build_uniform_dr_prompt(task.task_description, task.draft_policy, "40% slipped")
    p5 = build_physcap_prompt(task.task_description, task.draft_policy, "mass=0.4")
    for p in (p1, p2, p3, p4, p5):
        for param_name in ("heavy_mass_kg", "yield_force_n", "distractor_temp", "distractor_finish"):
            assert param_name not in p

    extracted = extract_python_code("Here you go:\n```python\ndef run(api):\n    pass\n```\n")
    assert extracted == "def run(api):\n    pass\n"

    mock = MockLLMClient(["```python\ndef run(api):\n    api.get_ee_pose()\n```"])
    cached = DiskCacheLLM(mock, cache_dir=tmp_path / "llm_cache")
    r1 = cached.complete("hello prompt", system="sys")
    r2 = cached.complete("hello prompt", system="sys")
    assert not r1.cached
    assert r2.cached
    assert r1.text == r2.text
    assert mock.stats["calls"] == 1
    assert cached.stats["calls"] == 2
    assert cached.stats["cached_hits"] == 1
    assert cached.stats["prompt_tokens"] == r1.prompt_tokens + r2.prompt_tokens
    assert cached.stats["completion_tokens"] == r1.completion_tokens + r2.completion_tokens


def test_disk_cache_llm_thread_safety(tmp_path: Path) -> None:
    mock = MockLLMClient(lambda prompt: f"resp:{prompt}")
    cached = DiskCacheLLM(mock, cache_dir=tmp_path / "thread_cache")
    prompts = [f"prompt-{i % 4}" for i in range(32)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: cached.complete(p, system="sys"), prompts))

    assert len(results) == 32
    assert all(r.text == f"resp:{p}" for p, r in zip(prompts, results))
    assert cached.stats["calls"] == 32
    assert cached.stats["cached_hits"] + mock.stats["calls"] == 32
    assert mock.stats["calls"] >= 4
    assert cached.stats["prompt_tokens"] == sum(r.prompt_tokens for r in results)
    assert cached.stats["completion_tokens"] == sum(r.completion_tokens for r in results)


def test_codex_cli_smoke_with_disk_cache(tmp_path: Path) -> None:
    codex_bin = Path("/home/wen/.local/bin/codex")
    if not codex_bin.exists():
        pytest.skip("codex binary not available")

    client = DiskCacheLLM(
        CodexCLIClient(model="gpt-6.1-sol", effort="low", timeout_s=120),
        cache_dir=tmp_path / "codex_cache",
    )
    prompt = (
        "Reply with ONLY a python code block defining `def run(api):` that calls "
        "`api.get_ee_pose()`. Do not inspect any local workspace files."
    )
    resp1 = client.complete(prompt)
    assert resp1.text
    assert not resp1.cached
    assert resp1.prompt_tokens > 0
    code = extract_python_code(resp1.text)
    check_policy_source(code)

    resp2 = client.complete(prompt)
    assert resp2.cached
    assert resp2.text == resp1.text
    assert client.inner.stats["calls"] == 1
