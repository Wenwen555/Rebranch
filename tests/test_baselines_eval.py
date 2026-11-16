"""Tests for §7 baselines, §9 evaluation metrics, runner, and table generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rebranch.baselines import BaselineResult
from rebranch.baselines.oneshot_calib import run_oneshot_calib
from rebranch.baselines.oracle_twin import run_oracle_twin
from rebranch.baselines.physcap_agent import run_physcap
from rebranch.baselines.single_twin import run_single_twin
from rebranch.baselines.twin_mpc import run_twin_mpc
from rebranch.baselines.uniform_dr import run_uniform_dr
from rebranch.eval.metrics import (
    compute_branch_discrimination_accuracy,
    compute_metrics,
    policy_has_sensory_branch,
    summarize_trace_for_llm,
    trace_used_probe,
)
from rebranch.exec.sandbox import run_policy
from rebranch.exec.trace import Trace
from rebranch.llm.client import MockLLMClient
from rebranch.sim.toy_backend import ToyBranchTask, ToyRetuneTask
from scripts.make_tables import generate_tables


_TOY_BRANCH_PROBED_POLICY = """\
GRIP_PROBE = 3.0  # [1.0, 20.0]
GRIP_FRAGILE = 3.0  # [1.0, 20.0]
GRIP_HEAVY = 16.5  # [1.0, 20.0]
WIDTH_THRESH = 0.051  # [0.02, 0.06]
LIFT_SPEED = 0.02  # [0.01, 0.20]

def run(api):
    api.zero_ee_wrench()
    p = api.get_object_pose("cup")["pos"]
    api.open_gripper(0.065)
    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)
    api.move_to([p[0], p[1], p[2]], speed=0.04)
    api.close_gripper(force=GRIP_PROBE)
    st = api.get_gripper_state()
    if st["width"] < WIDTH_THRESH:
        api.close_gripper(force=GRIP_FRAGILE)
    else:
        api.close_gripper(force=GRIP_HEAVY)
    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)
"""


def test_uniform_dr_learns_from_rejected_candidate() -> None:
    prompts = []
    rejected = 'def run(api):\n    api.open_gripper(0.065)\n'

    def respond(prompt: str) -> str:
        prompts.append(prompt)
        source = rejected if len(prompts) == 1 else _TOY_BRANCH_PROBED_POLICY
        return f"```python\n{source}```"

    result = run_uniform_dr(
        ToyBranchTask(), MockLLMClient(respond), seed=0,
        max_rollouts=100, max_llm_calls=2, twins_per_round=4, admission_n_eval=4,
    )
    assert len(prompts) == 2
    assert "Previous Rejected Attempt:" not in prompts[0]
    assert rejected in prompts[1]
    assert "Batch summary: 0/4 succeeded" in prompts[1]
    assert [p["accepted"] for p in result.program_snapshots] == [True, False, True]
    assert result.policy_source.strip() == _TOY_BRANCH_PROBED_POLICY.strip()
    assert result.n_rollouts <= 100


def test_metrics_and_branch_discrimination() -> None:
    task = ToyBranchTask()
    cases = task.hidden_test_set(seed=2026, n=20)

    # Non-branching draft policy on 2-mode task should get chance-level ~0.5 branch accuracy
    # and 100% missed_probe_rate.
    draft_traces = [
        run_policy(task.draft_policy, task.make_episode(th, seed=i), th, seed=i)
        for i, th in enumerate(cases)
    ]
    m_draft = compute_metrics(
        task,
        draft_traces,
        policy_sources=[task.draft_policy] * len(cases),
    )
    assert m_draft.missed_probe_rate == pytest.approx(1.0)
    assert m_draft.redundant_probe_rate == pytest.approx(0.0)
    assert m_draft.branch_discrimination_accuracy < 0.75

    # Crashed traces (e.g. conservative 20 N crushing fragile cups) must NOT get fake branch credit
    cons_traces = [
        run_policy(task.conservative_policy(), task.make_episode(th, seed=i), th, seed=i)
        for i, th in enumerate(cases)
    ]
    m_cons = compute_metrics(
        task,
        cons_traces,
        policy_sources=[task.conservative_policy()] * len(cases),
    )
    assert m_cons.safety_violation_rate > 0.2
    assert m_cons.branch_discrimination_accuracy < 0.75

    # Probing + branching policy gets 100% branch discrimination and 0 missed probe rate
    branch_traces = [
        run_policy(_TOY_BRANCH_PROBED_POLICY, task.make_episode(th, seed=i), th, seed=i)
        for i, th in enumerate(cases)
    ]
    m_branch = compute_metrics(
        task,
        branch_traces,
        policy_sources=[_TOY_BRANCH_PROBED_POLICY] * len(cases),
    )
    assert m_branch.zero_shot_success_rate == pytest.approx(1.0)
    assert m_branch.missed_probe_rate == pytest.approx(0.0)
    assert m_branch.sim_false_positive_rate is None
    assert m_branch.prediction_coverage == 0.0
    assert m_branch.branch_discrimination_accuracy == pytest.approx(1.0)

    # Non-branching task checks redundant_probe_rate
    retune_task = ToyRetuneTask()
    r_cases = retune_task.hidden_test_set(seed=2026, n=8)
    r_traces = [
        run_policy(retune_task.draft_policy, retune_task.make_episode(th, seed=i), th, seed=i)
        for i, th in enumerate(r_cases)
    ]
    m_retune = compute_metrics(
        retune_task, r_traces, policy_sources=[retune_task.draft_policy] * len(r_cases)
    )
    assert m_retune.redundant_probe_rate == pytest.approx(0.0)
    assert compute_branch_discrimination_accuracy(retune_task, r_traces) == pytest.approx(1.0)

    # Straight-line get_object_pose (e.g. draft_policy) is NOT counted as a sensory probe,
    # whereas an explicit ast.If conditioned on get_object_pose IS counted.
    straight_pose_src = (
        "def run(api):\n"
        '    p = api.get_object_pose("cup")["pos"]\n'
        "    api.open_gripper(0.065)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
    )
    branch_pose_src = (
        "def run(api):\n"
        '    q = api.get_object_pose("cup")["quat"]\n'
        "    if abs(q[2]) > 0.05:\n"
        "        api.open_gripper(0.065)\n"
        "    else:\n"
        "        api.open_gripper(0.055)\n"
    )
    assert not policy_has_sensory_branch(straight_pose_src)
    assert policy_has_sensory_branch(branch_pose_src)
    sp_trace = run_policy(straight_pose_src, task.make_episode(cases[0], seed=0), cases[0], seed=0)
    bp_trace = run_policy(branch_pose_src, task.make_episode(cases[0], seed=0), cases[0], seed=0)
    assert not trace_used_probe(sp_trace, policy_source=straight_pose_src)
    assert trace_used_probe(bp_trace, policy_source=branch_pose_src)
    assert compute_metrics(task, [bp_trace], policy_sources=[branch_pose_src]).missed_probe_rate == pytest.approx(0.0)
    assert compute_metrics(task, [sp_trace], policy_sources=[straight_pose_src]).missed_probe_rate == pytest.approx(1.0)
    assert compute_metrics(retune_task, [sp_trace], policy_sources=[straight_pose_src]).redundant_probe_rate == pytest.approx(0.0)


def test_all_six_baselines_on_toy_branch_no_theta_leak() -> None:
    task = ToyBranchTask()
    cases = task.hidden_test_set(seed=2026, n=8)

    def _responder(prompt: str) -> str:
        for secret in ("heavy_mass_kg", "yield_force_n", "distractor_temp", "distractor_finish"):
            assert secret not in prompt, f"Leaked theta parameter {secret!r} into prompt!"
        if "get_mass()" in prompt:
            return (
                "```python\n"
                "def run(api):\n"
                "    api.zero_ee_wrench()\n"
                "    _m = api.get_mass()\n"
                "    _k = api.get_stiffness()\n"
                '    p = api.get_object_pose("cup")["pos"]\n'
                "    api.open_gripper(0.065)\n"
                "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
                "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
                "    api.close_gripper(force=3.0 if _k < 350.0 else 16.5)\n"
                "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
                "```\n"
            )
        return f"```python\n{_TOY_BRANCH_PROBED_POLICY}```\n"

    llm = MockLLMClient(_responder)

    # 1. Single-Twin
    st_res = run_single_twin(task, llm, seed=0, max_rollouts=10, max_llm_calls=3)
    assert st_res.n_rollouts <= 10
    assert st_res.dev_llm_calls == 0  # draft already passes nominal

    # 2. Uniform-DR
    udr_res = run_uniform_dr(task, llm, seed=0, max_rollouts=40, max_llm_calls=3)
    assert udr_res.n_rollouts <= 40
    assert udr_res.dev_llm_calls <= 3

    # 3. PhysCaP
    pc_res = run_physcap(task, llm, max_dev_llm_calls=1)
    assert pc_res.agent is not None
    pc_traces = []
    pc_rt_calls = []
    for i, th in enumerate(cases[:4]):
        tr, meta = pc_res.agent.evaluate_episode(task, th, seed=i)
        pc_traces.append(tr)
        pc_rt_calls.append(meta["runtime_llm_calls"])
    assert sum(pc_rt_calls) == len(pc_traces)
    assert any(c.name == "get_mass" for tr in pc_traces for c in tr.calls)

    # 4. Twin-MPC
    mpc_res = run_twin_mpc(task, llm, seed=0, max_rollouts=48, max_llm_calls=2, n_twins=8)
    assert mpc_res.n_rollouts <= 48
    assert mpc_res.dev_llm_calls <= 2
    assert mpc_res.agent is not None
    tr_mpc, meta_mpc = mpc_res.agent.evaluate_episode(task, cases[0], seed=0)
    assert isinstance(tr_mpc, Trace)
    assert meta_mpc["runtime_llm_calls"] == 0

    # 5. 1-Shot Sim-Calib
    os_res = run_oneshot_calib(task, llm, seed=0, max_rollouts=48, max_llm_calls=2, n_twins=8)
    assert os_res.n_rollouts <= 48
    assert os_res.dev_llm_calls <= 2
    assert os_res.agent is not None
    tr_os, meta_os = os_res.agent.evaluate_episode(task, cases[0], seed=0)
    assert isinstance(tr_os, Trace)
    assert meta_os["runtime_llm_calls"] == 0

    # 6. Oracle-Twin
    ora_res = run_oracle_twin(task)
    assert ora_res.agent is not None
    ora_ok = sum(
        ora_res.agent.evaluate_episode(task, th, seed=i)[0].success
        for i, th in enumerate(cases)
    )
    assert ora_ok == len(cases)

    # Verify summarize_trace_for_llm leaks no theta keys
    summary = summarize_trace_for_llm(tr_mpc)
    assert "heavy_mass_kg" not in summary
    assert "yield_force_n" not in summary


def test_make_tables_generation(tmp_path: Path) -> None:
    import json

    in_dir = tmp_path / "m4"
    in_dir.mkdir(parents=True)
    rows = [
        {
            "task": "t2_cup",
            "method": "rebranch",
            "variant": "full",
            "split": "hidden",
            "seed": 0,
            "metrics": {
                "zero_shot_success_rate": 0.94,
                "sim_false_positive_rate": 0.06,
                "safety_violation_rate": 0.02,
                "redundant_probe_rate": 0.0,
                "missed_probe_rate": 0.0,
                "branch_discrimination_accuracy": 0.96,
                "dev_llm_calls": 2,
                "dev_tokens": 3200,
                "runtime_llm_calls": 0.0,
                "runtime_tokens": 0.0,
                "mean_exec_time_s": 0.35,
            },
        },
        {
            "task": "t2_cup",
            "method": "rebranch",
            "variant": "random_ce",
            "split": "hidden",
            "seed": 0,
            "metrics": {
                "zero_shot_success_rate": 0.72,
                "sim_false_positive_rate": 0.28,
                "safety_violation_rate": 0.14,
                "redundant_probe_rate": 0.0,
                "missed_probe_rate": 0.10,
                "branch_discrimination_accuracy": 0.78,
                "dev_llm_calls": 3,
                "dev_tokens": 4100,
                "runtime_llm_calls": 0.0,
                "runtime_tokens": 0.0,
                "mean_exec_time_s": 0.36,
            },
        },
    ]
    (in_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "tables"
    paths = generate_tables([in_dir], out_dir)
    assert paths["main_md"].exists()
    assert paths["main_csv"].exists()
    assert paths["ablation_md"].exists()
    assert paths["ablation_csv"].exists()
    assert "94.0" in paths["main_md"].read_text(encoding="utf-8")
    assert "random_ce" in paths["ablation_md"].read_text(encoding="utf-8")


def test_tables_pair_aggregate_support(tmp_path: Path) -> None:
    import csv
    import json

    rows = [
        {"task": "shared", "method": "rebranch", "variant": "full", "split": "hidden", "seed": 0, "metrics": {"zero_shot_success_rate": 0.2}},
        {"task": "shared", "method": "rebranch", "variant": "random_ce", "split": "hidden", "seed": 0, "metrics": {"zero_shot_success_rate": 0.4}},
        {"task": "extra", "method": "rebranch", "variant": "full", "split": "hidden", "seed": 0, "metrics": {"zero_shot_success_rate": 1.0}},
        {"task": "shared", "method": "uniform_dr", "variant": "full", "split": "hidden", "seed": 0, "metrics": {"zero_shot_success_rate": 0.1}},
    ]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    paths = generate_tables([tmp_path], tmp_path / "tables")
    for key in ("main_csv", "ablation_csv"):
        with paths[key].open() as fh:
            summaries = list(csv.DictReader(fh))
        assert all(int(r["n_runs"]) == 1 for r in summaries)
        ours = next(r for r in summaries if r.get("method") == "rebranch" or r.get("variant") == "full")
        assert float(ours["zero_shot_success_rate_mean"]) == pytest.approx(0.2)


def test_runner_suite_budget_parity_and_ablations(tmp_path: Path) -> None:
    from rebranch.eval.runner import ABLATION_VARIANTS, run_benchmark_suite

    def _responder(prompt: str) -> str:
        for secret in ("heavy_mass_kg", "yield_force_n", "distractor_temp", "distractor_finish"):
            assert secret not in prompt
        if "get_mass()" in prompt:
            return (
                "```python\n"
                "def run(api):\n"
                "    api.zero_ee_wrench()\n"
                "    _m = api.get_mass()\n"
                "    _k = api.get_stiffness()\n"
                '    p = api.get_object_pose("cup")["pos"]\n'
                "    api.open_gripper(0.065)\n"
                "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
                "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
                "    api.close_gripper(force=3.0 if _k < 350.0 else 16.5)\n"
                "    api.move_to([p[0], p[1], p[2] + 0.12], speed=0.02)\n"
                "```\n"
            )
        return f"```python\n{_TOY_BRANCH_PROBED_POLICY}```\n"

    llm = MockLLMClient(_responder)
    out_dir = tmp_path / "suite_out"
    ablations = [v for v in ABLATION_VARIANTS if v != "full"]
    run_benchmark_suite(
        tasks=["toy_branch"],
        methods=[
            "rebranch",
            "single_twin",
            "uniform_dr",
            "physcap",
            "twin_mpc",
            "oneshot_calib",
            "oracle_twin",
        ],
        llm=llm,
        ablations=ablations,
        seeds=[0],
        workers=1,
        n_hidden=10,
        n_ood=4,
        max_rollouts=600,
        max_llm_calls=4,
        max_rounds=2,
        n_ce_samples=40,
        output_dir=out_dir,
        match_rebranch_budget=True,
    )

    assert (out_dir / "runs.jsonl").exists()
    assert (out_dir / "runs.csv").exists()
    assert list((out_dir / "rounds").glob("*.jsonl"))
    assert list((out_dir / "snapshots").glob("*_final.py"))

    tables_dir = tmp_path / "tables"
    tpaths = generate_tables([out_dir], tables_dir)
    main_txt = tpaths["main_md"].read_text(encoding="utf-8")
    abl_txt = tpaths["ablation_md"].read_text(encoding="utf-8")
    for m in ("rebranch", "single_twin", "uniform_dr", "physcap", "twin_mpc", "oneshot_calib", "oracle_twin"):
        assert m in main_txt
    for a in ablations:
        assert a in abl_txt

