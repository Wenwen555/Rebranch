"""M3 Verification suite for PiPER-X manipulation tasks (T1-T8)."""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pytest

from rebranch.api.primitives import PRIMITIVES
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.sandbox import check_policy_source, run_policy
from rebranch.sim.backend import SNAPSHOT_KEYS, EpisodeResult

M3_CORE_TASKS = (
    "t1_eccentric_box",
    "t2_cup",
    "t3_push_can",
    "t4_cabinet",
    "t5_drawer",
    "t6_valve",
    "t7_peg",
    "t8_card",
)


def test_task_registry_and_zero_sensor_leak() -> None:
    from rebranch.tasks import TASKS, make_task

    assert set(M3_CORE_TASKS).issubset(TASKS)
    for idx, name in enumerate(M3_CORE_TASKS, start=1):
        t = make_task(name)
        t_alias = make_task(f"t{idx}")
        assert t.name == name
        assert t_alias.name == name
        assert t.obs_scale
        assert t.obs_api_map
        check_policy_source(t.draft_policy)
        check_policy_source(t.conservative_policy())
        check_policy_source(t.oracle_policy(t.param_space.nominal()))

        # Confirm each task includes 1-2 distractor parameters
        distractors = [p.name for p in t.param_space if p.name.startswith("distractor_")]
        assert 1 <= len(distractors) <= 2

        # Confirm no <force> or <torque> sensor or slip_distance in task source
        src = inspect.getsource(type(t))
        assert "<force" not in src
        assert "<torque" not in src
        assert "slip_distance" not in src

    # Confirm no <force>/<torque> in piperx.xml and no slip_distance in rebranch/tasks/
    tasks_dir = Path(__file__).resolve().parents[1] / "rebranch" / "tasks"
    for p in tasks_dir.rglob("*.py"):
        txt = p.read_text(encoding="utf-8")
        assert "slip_distance" not in txt
        assert "<force" not in txt
        assert "<torque" not in txt
    piperx_xml = tasks_dir / "assets" / "piperx" / "piperx.xml"
    xml_txt = piperx_xml.read_text(encoding="utf-8")
    assert "<force" not in xml_txt
    assert "<torque" not in xml_txt


def test_determinism_and_snapshot_keys() -> None:
    from rebranch.tasks import make_task

    for name in ("t2_cup", "t4_cabinet"):
        task = make_task(name)
        theta = task.param_space.nominal()
        tr1 = run_policy(task.draft_policy, task.make_episode(theta, seed=11), theta, seed=11)
        tr2 = run_policy(task.draft_policy, task.make_episode(theta, seed=11), theta, seed=11)
        assert tr1.success == tr2.success
        assert tr1.score == pytest.approx(tr2.score)
        assert tr1.lines == tr2.lines
        assert [c.obs for c in tr1.calls] == [c.obs for c in tr2.calls]
        for call in tr1.calls:
            assert tuple(call.obs.keys()) == SNAPSHOT_KEYS


def test_distractors_are_physics_invariant() -> None:
    from rebranch.tasks import make_task

    for name in M3_CORE_TASKS:
        task = make_task(name)
        base = task.param_space.nominal()
        perturbed = dict(base)
        for p in task.param_space:
            if p.name.startswith("distractor_"):
                if hasattr(p, "hi"):
                    perturbed[p.name] = p.hi
                else:
                    perturbed[p.name] = p.choices[-1]
        tr_base = run_policy(task.draft_policy, task.make_episode(base, seed=5), base, seed=5)
        tr_pert = run_policy(
            task.draft_policy, task.make_episode(perturbed, seed=5), perturbed, seed=5
        )
        assert tr_base.success == tr_pert.success
        assert tr_base.score == pytest.approx(tr_pert.score, abs=1e-6)


def _run_oracle_worker(args: tuple[str, dict, int]) -> object:
    task_name, th, seed = args
    from rebranch.tasks import make_task

    task = make_task(task_name)
    return run_policy(task.oracle_policy(th), task.make_episode(th, seed=seed), th, seed=seed)


@pytest.mark.parametrize("task_name", M3_CORE_TASKS)
def test_task_nominal_oracle_conservative_and_ood(task_name: str) -> None:
    from rebranch.tasks import make_task

    task = make_task(task_name)
    ev = Evaluator(task, workers=8)
    try:
        nom = task.param_space.nominal()
        tr_nom = ev.run(task.draft_policy, nom, seed=0)
        assert tr_nom.success, f"{task_name} nominal failed: {tr_nom.failure}"
        assert tr_nom.score > 0.0

        hidden = task.hidden_test_set(seed=2026, n=50)
        assert len(hidden) == 50

        cons_sr = ev.success_rate(task.conservative_policy(), hidden, seed=100)
        assert cons_sr < 0.50, f"{task_name} conservative={cons_sr:.2f} must be < 0.50"

        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
            oracle_traces = list(
                ex.map(_run_oracle_worker, [(task_name, th, 100 + i) for i, th in enumerate(hidden)])
            )
        oracle_sr = sum(1 for t in oracle_traces if t.success) / len(oracle_traces)
        assert oracle_sr >= 0.85, f"{task_name} oracle={oracle_sr:.2f} must be >= 0.85"

        # OOD sanity check
        ood = task.ood_test_set(seed=3026, n=10)
        assert len(ood) == 10
        with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
            ood_traces = list(
                ex.map(_run_oracle_worker, [(task_name, th, 500 + i) for i, th in enumerate(ood)])
            )
        assert all(isinstance(t.score, float) and -1.0 <= t.score <= 1.0 for t in ood_traces)
        assert all(isinstance(task.latent_mode(th), str) and task.latent_mode(th) for th in ood)
    finally:
        ev.close()


def test_t2_cup_crush_hook_and_proprioceptive_separability() -> None:
    from rebranch.tasks import make_task

    task = make_task("t2_cup")
    probe_policy = (
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("cup")["pos"]\n'
        "    api.open_gripper(0.068)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
        "    api.close_gripper(force=3.0)\n"
        "    api.get_gripper_state()\n"
        "    api.move_to([p[0], p[1], p[2] + 0.060], speed=0.05)\n"
        "    api.get_ee_wrench()\n"
        "    api.get_gripper_state()\n"
    )

    th_empty = task.param_space.nominal()
    th_empty["cup_type"] = "empty"
    tr_empty = run_policy(probe_policy, task.make_episode(th_empty, seed=3), th_empty, seed=3)

    th_sand = task.param_space.nominal()
    th_sand["cup_type"] = "sand"
    th_sand["sand_mass_kg"] = 0.45
    tr_sand = run_policy(probe_policy, task.make_episode(th_sand, seed=3), th_sand, seed=3)

    # v1's >=8 mm gap was an impact transient, not static stiffness.
    # Check the settled soft-contact deformation against exact MuJoCo geometry;
    # sensor-noise separability is tested by repeated differential measurements.
    clean_widths = []
    for th in (th_empty, th_sand):
        ep = task.make_episode(th, seed=3)
        ep._noise.enabled = False
        tr = run_policy(probe_policy, ep, th, seed=3)
        clean_widths.append(tr.calls[6].obs["grip_width"])
    assert .001 < clean_widths[1] - clean_widths[0] < .004

    fz_empty = tr_empty.calls[8].obs["fz"]
    fz_sand = tr_sand.calls[8].obs["fz"]
    slipped_sand = tr_sand.calls[9].obs["grip_grasped"] < 0.5
    held_empty = tr_empty.calls[9].obs["grip_grasped"] > 0.5
    assert held_empty
    assert (fz_empty - fz_sand >= 1.5) or slipped_sand

    # Crush hook verification
    crush_tr = run_policy(task.conservative_policy(), task.make_episode(th_empty, seed=0), th_empty, seed=0)
    assert not crush_tr.success
    assert crush_tr.failure == "crushed"
    assert crush_tr.safety_violation == "crushed"

    # Anti-retune verification: cup_geom.priority=1 prevents pad friction=1.2 override,
    # so even the lightest/highest-friction sand cup slips at sub-4N constant grip force.
    light_sand = task.param_space.nominal()
    light_sand["cup_type"] = "sand"
    light_sand["sand_mass_kg"] = 0.30
    light_sand["friction"] = 0.34
    sub4n_policy = (
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    p = api.get_object_pose("cup")["pos"]\n'
        "    api.open_gripper(0.068)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
        "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
        "    api.close_gripper(force=3.5)\n"
        "    api.move_to([p[0], p[1], p[2] + 0.11], speed=0.04)\n"
    )
    slip_tr = run_policy(sub4n_policy, task.make_episode(light_sand, seed=0), light_sand, seed=0)
    assert not slip_tr.success
    assert slip_tr.failure in ("slipped", "not_lifted")


def test_t4_cabinet_proprioceptive_separability() -> None:
    from rebranch.tasks import make_task

    task = make_task("t4_cabinet")
    probe_policy = (
        "def run(api):\n"
        "    api.zero_ee_wrench()\n"
        '    h = api.get_object_pose("handle")["pos"]\n'
        "    api.open_gripper(0.06)\n"
        "    api.move_to([h[0] - 0.06, h[1], h[2]], speed=0.08)\n"
        "    api.move_to([h[0], h[1], h[2]], speed=0.05)\n"
        "    api.close_gripper(force=15.0)\n"
        "    api.get_ee_pose()\n"
        '    api.move_compliant([h[0] - 0.035, h[1], h[2]], free_axes="yzXYZ", speed=0.03)\n'
        "    api.get_ee_pose()\n"
        "    api.get_ee_wrench()\n"
    )

    deltas: dict[str, tuple[float, float, float]] = {}
    for topo in ("left_hinge", "right_hinge", "bottom_flap", "drawer"):
        th = task.param_space.nominal()
        th["topology"] = topo
        tr = run_policy(probe_policy, task.make_episode(th, seed=2), th, seed=2)
        p_before = tr.calls[6].obs
        p_after = tr.calls[8].obs
        dy = p_after["ee_y"] - p_before["ee_y"]
        dz = p_after["ee_z"] - p_before["ee_z"]
        dx = p_after["ee_x"] - p_before["ee_x"]
        deltas[topo] = (dx, dy, dz)

    # All 4 topologies cleanly separated by compliant deflection (dy, dz)
    assert deltas["left_hinge"][1] > 0.004
    assert deltas["right_hinge"][1] < -0.004
    assert deltas["bottom_flap"][2] < -0.004
    assert abs(deltas["drawer"][1]) < 0.003 and abs(deltas["drawer"][2]) < 0.003
    assert deltas["drawer"][0] < -0.015


def test_t1_eccentric_box_conflict_and_probe_separability() -> None:
    from rebranch.core.conflict import distinguishing_observable, incompatible_pairs, per_case_solutions
    from rebranch.core.lifting import lift_constants
    from rebranch.core.retune import Retuner
    from rebranch.exec.evaluator import Evaluator
    from rebranch.tasks import make_task

    task = make_task("t1_eccentric_box")
    th0 = task.param_space.nominal()
    th_neg = dict(th0)
    th_neg["com_x"] = -0.032
    th_pos = dict(th0)
    th_pos["com_x"] = 0.032
    cases = [th0, th_neg, th_pos]

    ev = Evaluator(task, workers=8)
    try:
        consts = lift_constants(task.draft_policy)
        retuner = Retuner(ev, max_iters=8, popsize=8, seed=0)
        sols = per_case_solutions(retuner, task.draft_policy, consts, cases)
        assert all(sol.solvable for sol in sols), f"Expected all cases individually solvable, got {[s.score for s in sols]}"

        joint = retuner.optimize(task.draft_policy, consts, cases)
        assert not joint.success_all

        base_traces = ev.run_many(task.draft_policy, cases)
        pairs = incompatible_pairs(ev, task.draft_policy, consts, cases, sols, retuner=retuner)
        assert pairs, "Expected incompatible pairs across eccentric cases"

        ranked = distinguishing_observable(
            base_traces,
            pairs,
            task.obs_scale,
            task.obs_api_map,
            nominal_trace=base_traces[0],
        )
        assert ranked, "Expected non-empty distinguishing observables"
        lift_ty = [r for r in ranked if r.key == "ty" and r.call_index == 7]
        assert lift_ty, f"Expected ty on lift call (index 7), got {ranked[:4]}"
        assert lift_ty[0].opposite_sign is True
        assert ranked[0].key == "ty" and ranked[0].call_index == 7
    finally:
        ev.close()
