from __future__ import annotations

from rebranch.exec.sandbox import check_policy_source
from rebranch.tasks.t9_supported_tray import (
    CAN_MASS_BOUNDS,
    SLOT_CENTER_TOL,
    T9SupportedTrayTask,
    enumerate_static_plans,
)


def test_t9_public_target_and_shared_transport_are_explicit():
    task = T9SupportedTrayTask()
    assert SLOT_CENTER_TOL == 0.010
    assert "68 mm diameter" in task.task_description
    assert "10 mm radial distance" in task.task_description
    assert "1.0 s window" in task.task_description
    assert "0.200 m" in task.task_description
    sources = [task.conservative_policy(), task.probe_policy(), task.feedback_policy()]
    for source in sources:
        check_policy_source(source)
        assert "def place_held(api" in source
        assert "target_center_z + 0.020" in source
        assert "e[0] - o[0]" in source
        assert "get_object_pose(slot)" in source
        assert "-0.006" in source


def test_t9_static_domain_corners_have_an_assignment_and_order():
    lo, hi = CAN_MASS_BOUNDS
    for mass_a in (lo, hi):
        for mass_b in (lo, hi):
            plans = enumerate_static_plans(mass_a, mass_b)
            assert plans
            assert all(set(plan["assignment"]) == {"can_a", "can_b"} for plan in plans)
            assert all(len(plan["order"]) == 2 for plan in plans)
    biased = enumerate_static_plans(0.04, 0.30)
    assert any(plan["assignment"]["can_b"] == "inner" for plan in biased)


def test_t9_probe_is_per_object_paired_continuous_and_feedback_checks_while_held():
    task = T9SupportedTrayTask()
    probe = task.probe_policy()
    feedback = task.feedback_policy()
    assert "api.zero_ee_wrench()" in probe
    assert probe.count("get_ee_wrench(\"world\")") == 2
    assert "mass_kg = abs(load_fz - base_fz) / 9.81" in probe
    assert "first_slot = \"slot_inner\" if first == inner" in probe
    assert "lower_trial_while_held" in feedback
    trial = feedback[feedback.index("def lower_trial_while_held"):feedback.index("def run(api):")]
    assert trial.index("get_object_pose(\"tray\")") < trial.index("api.open_gripper")
    assert "get_object_pose(\"tray\")[\"quat\"]" in feedback
