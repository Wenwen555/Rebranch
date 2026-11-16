"""Unit and M0 end-to-end tests for ReBranch core algorithm modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rebranch.api.primitives import RobotAPI
from rebranch.core import (
    ConstSpec,
    FailureSignature,
    ReBranchConfig,
    Retuner,
    build_branch_prompt,
    distinguishing_observable,
    failure_signature,
    find_counterexamples,
    first_divergence,
    hoist_inline_literals,
    incompatible_pairs,
    instantiate,
    lift_constants,
    per_case_solutions,
    run_rebranch,
    sensitivity_screen,
    synthesize_branch,
    unsat_core,
    verify_branch,
)
from rebranch.exec import Evaluator, check_policy_source, run_policy
from rebranch.exec.trace import CallRecord, Trace
from rebranch.llm import MockLLMClient
from rebranch.sim import (
    SNAPSHOT_KEYS,
    Continuous,
    Discrete,
    Episode,
    EpisodeResult,
    ParamSpace,
    TaskBackend,
    ToyBranchTask,
    ToyRetuneTask,
)


def _make_dummy_trace(
    *,
    theta: dict[str, Any] | None = None,
    calls_obs: list[dict[str, float]] | None = None,
    success: bool = False,
    score: float = -0.5,
    failure: str | None = "slipped",
    lines: list[int] | None = None,
) -> Trace:
    calls: list[CallRecord] = []
    for idx, partial_obs in enumerate(calls_obs or []):
        obs = {k: 0.0 for k in SNAPSHOT_KEYS}
        obs.update(partial_obs)
        calls.append(
            CallRecord(
                index=idx,
                name="move_compliant" if idx == 0 else "move_to",
                lineno=10 + idx,
                occurrence=1,
                args=(),
                kwargs={},
                result=None,
                obs=obs,
            )
        )
    return Trace(
        theta=theta or {},
        seed=0,
        calls=calls,
        lines=lines or [10, 11],
        success=success,
        score=score,
        failure=failure,
        failure_call=(len(calls) - 1) if (not success and calls) else None,
    )


def test_sensitivity_screen_filters_distractors_and_fallback() -> None:
    rng = np.random.default_rng(123)
    for task, expected_distractors in (
        (ToyRetuneTask(), {"distractor_hue", "distractor_label"}),
        (ToyBranchTask(), {"distractor_temp", "distractor_finish"}),
    ):
        with Evaluator(task, workers=1) as ev:
            active = sensitivity_screen(
                ev, task.draft_policy, task.param_space, rng, n_base=8
            )
        assert active
        for d in expected_distractors:
            assert d not in active

    # Fallback: if a policy always succeeds regardless of theta, retain all parameters.
    always_ok_src = "def run(api):\n    api.zero_ee_wrench()\n"

    class _AlwaysOKTask(ToyRetuneTask):
        def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
            ep = super().make_episode(theta, seed=seed)
            ep.evaluate = lambda: EpisodeResult(success=True, score=0.5)  # type: ignore[method-assign]
            return ep

    with Evaluator(_AlwaysOKTask(), workers=1) as ev:
        retained = sensitivity_screen(
            ev, always_ok_src, _AlwaysOKTask().param_space, rng, n_base=4
        )
    assert set(retained) == set(_AlwaysOKTask().param_space.names)


def test_signature_divergence_and_mirror_symmetry() -> None:
    obs_scale = {k: 0.1 for k in SNAPSHOT_KEYS}
    nom = _make_dummy_trace(
        calls_obs=[{"fx": 0.0, "ty": 0.0}, {"fz": -1.0}],
        success=True,
        score=0.5,
        failure=None,
    )
    pos_x = _make_dummy_trace(
        calls_obs=[{"fx": 0.6, "ty": 0.0}, {"fz": -1.0}],
        success=False,
        score=-0.6,
        failure="jammed",
    )
    neg_x = _make_dummy_trace(
        calls_obs=[{"fx": -0.6, "ty": 0.0}, {"fz": -1.0}],
        success=False,
        score=-0.6,
        failure="jammed",
    )
    idx_p, devs_p = first_divergence(pos_x, nom, obs_scale, delta=3.0)
    idx_n, devs_n = first_divergence(neg_x, nom, obs_scale, delta=3.0)
    assert idx_p == 0
    assert idx_n == 0
    assert devs_p["fx"] == pytest.approx(6.0)
    assert devs_n["fx"] == pytest.approx(-6.0)

    sig_pos = failure_signature(pos_x, nom, obs_scale, delta=3.0)
    sig_neg = failure_signature(neg_x, nom, obs_scale, delta=3.0)
    assert sig_pos != sig_neg
    assert sig_pos.lineno == sig_neg.lineno
    assert sig_pos.occurrence == sig_neg.occurrence
    assert sig_pos.failure_type == sig_neg.failure_type
    assert sig_pos.signs[0] == ("fx", 1)
    assert sig_neg.signs[0] == ("fx", -1)
    assert len(sig_pos.signs) <= 2


def test_counterexamples_grouping_shrinking_and_ranking() -> None:
    task = ToyBranchTask()
    # High-force policy: passes heavy, crushes fragile cups.
    high_force_src = (
        "GRIP_FORCE = 16.0  # [1.0, 20.0]\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("cup")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
    )
    # Nominal is fragile (yield_force=5.5), so nominal crushes too; use a fragile-safe
    # nominal trace as reference for divergence.
    nom_theta = task.param_space.nominal()
    with Evaluator(task, workers=1) as ev:
        nom_trace = ev.run(task.draft_policy, nom_theta, seed=0)
        rng = np.random.default_rng(7)
        ces, stats = find_counterexamples(
            ev,
            high_force_src,
            task.param_space,
            nom_trace,
            task.obs_scale,
            rng,
            n_samples=60,
            k_max=4,
            n_bisect=6,
        )
    assert len(ces) >= 1
    assert len(ces) <= 4
    assert stats["n_failures"] > 0
    assert stats["n_bisect_rollouts"] > 0
    assert all(ce.signature.failure_type == "crushed" for ce in ces)
    # Ranking should be monotonically non-increasing by group_size * severity
    scores = [ce.group_size * ce.severity for ce in ces]
    assert scores == sorted(scores, reverse=True)


def test_lifting_roundtrip_and_hoist_inline_literals() -> None:
    src = (
        '"""Test policy."""\n'
        "GRIP_FORCE = 3.5  # [1.0, 20.0]\n"
        'MODE = "a"  # {"a", "b"}\n'
        "BIAS = -0.02  # [-0.1, 0.1]\n\n"
        "def run(api):\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
    )
    specs = lift_constants(src)
    assert [s.name for s in specs] == ["GRIP_FORCE", "MODE", "BIAS"]
    assert specs[0] == ConstSpec("GRIP_FORCE", "continuous", 3.5, 1.0, 20.0, None, 2)
    assert specs[1] == ConstSpec("MODE", "discrete", "a", None, None, ("a", "b"), 3)
    assert specs[2] == ConstSpec("BIAS", "continuous", -0.02, -0.1, 0.1, None, 4)

    new_vals = {"GRIP_FORCE": 12.25, "MODE": "b", "BIAS": 0.05}
    inst = instantiate(src, new_vals)
    assert "# [1.0, 20.0]" in inst
    assert '# {"a", "b"}' in inst
    assert "# [-0.1, 0.1]" in inst
    assert len(inst.splitlines()) == len(src.splitlines())
    specs2 = lift_constants(inst)
    assert {s.name: s.initial for s in specs2} == new_vals

    # Hoist inline literals
    inline_src = (
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        "    api.move_to([0.35, 0.0, 0.04], speed=0.08)\n"
        "    api.move_until_contact([0.35, 0.0, 0.01], speed=0.04, max_force=5.0, max_torque=0.7)\n"
        "    api.close_gripper(force=9.0)\n"
    )
    hoisted = hoist_inline_literals(inline_src)
    check_policy_source(hoisted)
    h_specs = lift_constants(hoisted)
    names = {s.name for s in h_specs}
    assert {
        "MOVE_TO_SPEED",
        "MOVE_UNTIL_CONTACT_SPEED",
        "MOVE_UNTIL_CONTACT_MAX_FORCE",
        "MOVE_UNTIL_CONTACT_MAX_TORQUE",
        "CLOSE_GRIPPER_FORCE",
    } <= names


def test_retuner_1d_cma_2d_and_discrete() -> None:
    task = ToyRetuneTask()
    cases = [
        {"mass_kg": 0.30, "friction": 0.50, "yield_force_n": 15.0, "distractor_hue": 0.5, "distractor_label": "a"},
        {"mass_kg": 0.25, "friction": 0.48, "yield_force_n": 14.0, "distractor_hue": 0.2, "distractor_label": "b"},
    ]

    # 1D continuous optimization (must NOT crash on cma 1D ValueError!)
    src_1d = (
        "GRIP_FORCE = 2.0  # [1.0, 20.0]\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("block")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
    )
    with Evaluator(task, workers=1) as ev:
        retuner = Retuner(ev, max_iters=10, margin_target=0.20, seed=1)
        res_1d = retuner.optimize(src_1d, lift_constants(src_1d), cases)
    assert res_1d.success_all
    assert 6.5 <= res_1d.best_values["GRIP_FORCE"] <= 13.5

    # 2D CMA-ES optimization on draft_policy
    with Evaluator(task, workers=1) as ev:
        retuner = Retuner(ev, max_iters=15, margin_target=0.25, seed=2)
        res_2d = retuner.optimize(
            task.draft_policy, lift_constants(task.draft_policy), cases
        )
    assert res_2d.success_all
    assert res_2d.worst_score >= 0.20

    # Discrete outer enumeration + early stop on initial constants
    src_disc = (
        'STRAT = "weak"  # {"weak", "strong"}\n'
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("block")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
        '    f = 2.0 if STRAT == "weak" else 10.0\n'
        "    api.close_gripper(force=f)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
    )
    with Evaluator(task, workers=1) as ev:
        retuner = Retuner(ev, max_iters=5, margin_target=0.10, seed=3)
        res_disc = retuner.optimize(src_disc, lift_constants(src_disc), cases)
    assert res_disc.success_all
    assert res_disc.best_values["STRAT"] == "strong"

    # Initial constants already pass with margin -> early stop at n_evals == 1
    src_good = instantiate(src_disc, {"STRAT": "strong"})
    with Evaluator(task, workers=1) as ev:
        retuner = Retuner(ev, max_iters=5, margin_target=0.05, seed=3)
        res_early = retuner.optimize(src_good, lift_constants(src_good), cases)
    assert res_early.success_all
    assert res_early.n_evals == 1


def test_conflict_unsat_core_pairs_and_distinguishing_observable() -> None:
    task = ToyBranchTask()
    fragile = task.param_space.nominal()
    heavy1 = dict(fragile, mode="heavy", heavy_mass_kg=0.42)
    heavy2 = dict(fragile, mode="heavy", heavy_mass_kg=0.46)
    cases = [fragile, heavy1, heavy2]
    consts = lift_constants(task.draft_policy)

    with Evaluator(task, workers=1) as ev:
        retuner = Retuner(ev, max_iters=10, margin_target=0.20, seed=4)
        sols = per_case_solutions(retuner, task.draft_policy, consts, cases)
        assert all(s.solvable for s in sols)

        core_idx = unsat_core(retuner, task.draft_policy, consts, cases, sols)
        assert len(core_idx) == 2
        assert 0 in core_idx  # fragile must be part of the conflict

        core_cases = [cases[i] for i in core_idx]
        core_sols = [sols[i] for i in core_idx]
        pairs = incompatible_pairs(
            ev, task.draft_policy, consts, core_cases, core_sols, retuner=retuner
        )
        assert pairs == [(0, 1)]
        core_traces = ev.run_many(task.draft_policy, core_cases)
        obs = distinguishing_observable(
            core_traces,
            pairs,
            task.obs_scale,
            obs_api_map=task.obs_api_map,
            nominal_trace=core_traces[0],
        )
        assert obs
        top_keys = {o.key for o in obs[:4]}
        assert "grip_width" in top_keys or "grip_grasped" in top_keys

    # Verify 2x weight for opposite signs across conflicting cases
    obs_scale = {k: 1.0 for k in SNAPSHOT_KEYS}
    t_a = _make_dummy_trace(calls_obs=[{"fx": 2.0, "fz": 5.0}], failure="a")
    t_b = _make_dummy_trace(calls_obs=[{"fx": -2.0, "fz": 1.8}], failure="b")
    # fx: |2 - (-2)| = 4.0 * 2 (opposite sign) = 8.0 > fz: |5 - 1.8| = 3.2 * 1 = 3.2
    ranked = distinguishing_observable([t_a, t_b], [(0, 1)], obs_scale)
    assert ranked[0].key == "fx"
    assert ranked[0].opposite_sign is True
    assert ranked[0].score == pytest.approx(8.0)

    # Multi-call trace: call 1 separates cleanly (>= 3.0 sigma, opposite sign on ty)
    # while call 3 has a larger post-crash spike on jd3 -> must pick call 1's ty.
    t_left = _make_dummy_trace(
        calls_obs=[
            {"ty": 0.0, "jd3": 0.0},
            {"ty": 2.0, "jd3": 0.2},
            {"ty": 0.5, "jd3": 1.0},
            {"ty": 0.1, "jd3": 20.0},
        ],
        failure="tilt_left",
        lines=[10, 11, 12, 13],
    )
    t_right = _make_dummy_trace(
        calls_obs=[
            {"ty": 0.0, "jd3": 0.0},
            {"ty": -2.0, "jd3": 0.2},
            {"ty": -0.5, "jd3": 1.0},
            {"ty": -0.1, "jd3": -1.0},
        ],
        failure="tilt_right",
        lines=[10, 11, 12, 13],
    )
    ranked_multi = distinguishing_observable([t_left, t_right], [(0, 1)], obs_scale)
    assert ranked_multi[0].key == "ty"
    assert ranked_multi[0].call_index == 1
    assert ranked_multi[0].opposite_sign is True


def test_branch_prompt_no_theta_leak_and_verify_retries() -> None:
    task = ToyBranchTask()
    fragile = task.param_space.nominal()
    heavy = dict(fragile, mode="heavy", heavy_mass_kg=0.45)
    core_cases = [fragile, heavy]

    with Evaluator(task, workers=1) as ev:
        core_traces = ev.run_many(task.draft_policy, core_cases)
        obs = distinguishing_observable(
            core_traces, [(0, 1)], task.obs_scale, obs_api_map=task.obs_api_map
        )
        prompt = build_branch_prompt(
            task.task_description,
            task.draft_policy,
            core_traces,
            [(0, 1)],
            obs,
            obs_api_map=task.obs_api_map,
        )
        for secret in (
            "heavy_mass_kg",
            "yield_force_n",
            "distractor_temp",
            "distractor_finish",
            "0.45",
        ):
            assert secret not in prompt

        # verify_branch rejects non-branching policy
        v_fail = verify_branch(ev, task.draft_policy, core_cases, [(0, 1)])
        assert not v_fail.passed
        assert not v_fail.paths_diverged

        # synthesize_branch retries when first candidate fails verify_branch, passes on 2nd
        good_branched = (
            "WIDTH_THRESH = 0.051  # [0.04, 0.06]\n"
            "FRAGILE_FORCE = 3.0  # [1.0, 6.0]\n"
            "HEAVY_FORCE = 16.5  # [8.0, 20.0]\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("cup")["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
            "    api.close_gripper(force=3.0)\n"
            "    st = api.get_gripper_state()\n"
            "    if st['width'] < WIDTH_THRESH:\n"
            "        api.close_gripper(force=FRAGILE_FORCE)\n"
            "        api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
            "    else:\n"
            "        api.close_gripper(force=HEAVY_FORCE)\n"
            "        api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
        )
        llm = MockLLMClient(
            [
                f"```python\n{task.draft_policy}\n```",
                f"```python\n{good_branched}\n```",
            ]
        )
        cand, ver = synthesize_branch(
            ev,
            llm,
            task.task_description,
            task.draft_policy,
            core_cases,
            core_traces,
            [(0, 1)],
            obs,
            max_retries=2,
        )
        assert ver.passed
        assert ver.paths_diverged
        assert llm.stats["calls"] == 2
        assert "WIDTH_THRESH" in cand

        # If a branched candidate has 0/N successes despite many constants, skip retune and retry LLM immediately.
        all_fail_many_consts = (
            "\n".join(f"C{i} = {0.1 * i:.1f}  # [0.0, 5.0]" for i in range(15))
            + "\n"
            + "def run(api):\n"
            + "    api.zero_ee_wrench()\n"
            + '    api.get_object_pose("cup")\n'
            + "    api.close_gripper(force=3.0)\n"
            + "    if api.get_gripper_state()['width'] < 0.051:\n"
            + "        api.move_to([0.35, 0.0, 0.04], speed=0.05)\n"
            + "    else:\n"
            + "        api.move_to([0.35, 0.0, 0.04], speed=0.05)\n"
        )
        rollouts_before = ev.n_rollouts
        llm2 = MockLLMClient(
            [
                f"```python\n{all_fail_many_consts}\n```",
                f"```python\n{good_branched}\n```",
            ]
        )
        cand2, ver2 = synthesize_branch(
            ev,
            llm2,
            task.task_description,
            task.draft_policy,
            core_cases,
            core_traces,
            [(0, 1)],
            obs,
            retuner=Retuner(ev, max_iters=20),
            max_retries=2,
        )
        assert ver2.passed
        assert llm2.stats["calls"] == 2
        assert ev.n_rollouts - rollouts_before == 4  # 2 verify rollouts per attempt, zero CMA-ES blowup
        assert "WIDTH_THRESH" in cand2
        retry_prompt = llm2.history[1]["prompt"]
        assert "Previous Candidate:" in retry_prompt
        assert "get_gripper_state" in retry_prompt
        assert "get_object_pose" in retry_prompt
        assert "quat=[1.0000,0.0000,0.0000,0.0000]" in retry_prompt
        assert "w=" in retry_prompt
        for secret in ("heavy_mass_kg", "yield_force_n", "distractor_temp", "distractor_finish"):
            assert secret not in retry_prompt


def test_repair_retries_with_trace_feedback() -> None:
    # Task where draft_policy fails to open gripper (cannot be fixed by retuning alone)
    task = ToyRetuneTask()
    bad_draft = (
        "GRIP_FORCE = 10.0  # [1.0, 20.0]\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("block")["pos"]\n'
        "    api.move_to([p[0] + 0.15, p[1], p[2]], speed=0.05)\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
        "    api.move_to([p[0] + 0.15, p[1], p[2] + 0.12], speed=0.02)\n"
    )
    bad_repair = (
        "GRIP_FORCE = 10.0  # [1.0, 20.0]\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("block")["pos"]\n'
        "    api.move_to([p[0] + 0.10, p[1], p[2]], speed=0.05)\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
        "    api.move_to([p[0] + 0.10, p[1], p[2] + 0.12], speed=0.02)\n"
    )
    good_repair = (
        "GRIP_FORCE = 10.0  # [1.0, 20.0]\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("block")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
        "    api.close_gripper(force=GRIP_FORCE)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
    )
    llm = MockLLMClient(
        [
            f"```python\n{bad_repair}\n```",
            f"```python\n{good_repair}\n```",
        ]
    )
    cfg = ReBranchConfig(
        max_rounds=1,
        n_screen_base=4,
        n_ce_samples=12,
        k_max=2,
        n_bisect=2,
        retune_iters=5,
        max_branch_retries=2,
        admission_n_eval=8,
        seed=5,
    )
    with Evaluator(task, workers=1) as ev:
        res = run_rebranch(task, ev, llm, cfg, initial_source=bad_draft)
    assert res.llm_calls == 2
    assert any(r.action == "REPAIR" and r.admitted for r in res.rounds)
    assert "Verification Trace Feedback:" in llm.history[1]["prompt"]
    assert "Previous Candidate:" in llm.history[1]["prompt"]
    for secret in ("mass_kg", "yield_force_n", "distractor_hue", "distractor_label"):
        assert secret not in llm.history[1]["prompt"]


def test_m0_end_to_end_toy_retune_and_toy_branch_100_hidden() -> None:
    # 1. ToyRetuneTask: zero LLM calls, distractors screened out, >= 95% on 100 hidden
    retune_task = ToyRetuneTask()
    hidden_retune = retune_task.hidden_test_set(seed=2026, n=100)
    mock_fail_if_called = MockLLMClient(
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not be called"))
    )
    with Evaluator(retune_task, workers=1) as ev:
        draft_retune_rate = ev.success_rate(retune_task.draft_policy, hidden_retune)
        res_retune = run_rebranch(
            retune_task,
            ev,
            mock_fail_if_called,
            ReBranchConfig(
                max_rounds=3,
                n_screen_base=8,
                n_ce_samples=80,
                k_max=4,
                n_bisect=5,
                retune_iters=15,
                admission_n_eval=20,
                seed=10,
            ),
        )
        final_retune_rate = ev.success_rate(res_retune.final_source, hidden_retune)

    assert res_retune.llm_calls == 0
    assert "distractor_hue" not in res_retune.active_params
    assert "distractor_label" not in res_retune.active_params
    assert any(r.action == "RETUNE" for r in res_retune.rounds)
    assert final_retune_rate >= 0.95
    assert final_retune_rate > draft_retune_rate + 0.30

    # 2. ToyBranchTask: Mock-LLM branch synthesis, distractors screened out, >= 95% on 100 hidden
    branch_task = ToyBranchTask()
    hidden_branch = branch_task.hidden_test_set(seed=2026, n=100)

    branched_policy = (
        "WIDTH_THRESH = 0.051  # [0.04, 0.06]\n"
        "FRAGILE_FORCE = 3.0  # [1.0, 5.5]\n"
        "HEAVY_FORCE = 16.5  # [8.0, 20.0]\n"
        "LIFT_SPEED = 0.02  # [0.01, 0.20]\n\n"
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("cup")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
        "    api.close_gripper(force=3.0)\n"
        "    st = api.get_gripper_state()\n"
        "    if st['width'] < WIDTH_THRESH:\n"
        "        api.close_gripper(force=FRAGILE_FORCE)\n"
        "        api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        "    else:\n"
        "        api.close_gripper(force=HEAVY_FORCE)\n"
        "        api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
    )
    mock_branch_llm = MockLLMClient([f"```python\n{branched_policy}\n```"])

    with Evaluator(branch_task, workers=1) as ev:
        draft_branch_rate = ev.success_rate(branch_task.draft_policy, hidden_branch)
        # Retune-only baseline on draft_policy over sample cases
        sample_cases = branch_task.hidden_test_set(seed=77, n=12)
        retune_only_res = Retuner(ev, max_iters=12, seed=5).optimize(
            branch_task.draft_policy,
            lift_constants(branch_task.draft_policy),
            sample_cases,
        )
        retune_only_rate = ev.success_rate(retune_only_res.best_source, hidden_branch)

        res_branch = run_rebranch(
            branch_task,
            ev,
            mock_branch_llm,
            ReBranchConfig(
                max_rounds=3,
                n_screen_base=8,
                n_ce_samples=80,
                k_max=4,
                n_bisect=5,
                retune_iters=12,
                admission_n_eval=20,
                seed=11,
            ),
        )
        final_branch_rate = ev.success_rate(res_branch.final_source, hidden_branch)

    assert "distractor_temp" not in res_branch.active_params
    assert "distractor_finish" not in res_branch.active_params
    assert any(r.action == "BRANCH" for r in res_branch.rounds)
    assert res_branch.llm_calls >= 1
    for entry in mock_branch_llm.history:
        for secret in (
            "heavy_mass_kg",
            "yield_force_n",
            "distractor_temp",
            "distractor_finish",
        ):
            assert secret not in entry["prompt"]
    assert final_branch_rate >= 0.95
    assert final_branch_rate > draft_branch_rate + 0.35
    assert final_branch_rate > retune_only_rate + 0.35


def test_rebranch_ablation_toggles_smoke() -> None:
    task = ToyRetuneTask()
    llm = MockLLMClient([f"```python\n{task.oracle_policy({'mass_kg': 0.30, 'friction': 0.48, 'yield_force_n': 15.0})}\n```"])
    for kwargs in (
        {"random_ce": True},
        {"worst_ce": True},
        {"no_retune_or_branch": True},
        {"no_lifting": True},
        {"no_branch_verify": True},
    ):
        cfg = ReBranchConfig(
            max_rounds=1,
            n_screen_base=2,
            n_ce_samples=15,
            k_max=2,
            n_bisect=2,
            retune_iters=4,
            admission_n_eval=6,
            **kwargs,
        )
        with Evaluator(task, workers=1) as ev:
            res = run_rebranch(task, ev, llm, cfg)
            assert res.final_source
