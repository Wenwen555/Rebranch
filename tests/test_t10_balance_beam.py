"""T10 registration, physical feasibility, support and measurement contracts."""

from __future__ import annotations

import json
import math

import pytest

from rebranch.exec.sandbox import run_policy
from rebranch.sim.kinder_scene import (
    BLOCK_HALF_SIZE,
    BLOCK_NAMES,
    ZONE_HALF_Y,
    ZONE_X,
    masses_by_block,
    nonstack_feasibility_witness,
)
from rebranch.tasks import TASKS, make_task
from rebranch.tasks.hidden_mass_balance_beam import _MASS_MEASUREMENT_DOC, _step_state


def _assert_public_feedback(feedback, diagnostics):
    """Check the public allowlist against the existing terminal score, without re-evaluation."""
    flags = {"all_footprints_in_2d_regions", "all_support_chains_reach_beam",
             "beam_stable_throughout_hold", "released_throughout_hold",
             "withdrawn_throughout_hold"}
    assert set(feedback) == flags | {"protocol_version", "hold_max_tilt_deg", "terminal_block_footprints"}
    assert feedback["protocol_version"] == "t10-terminal-feedback-v1"
    for name in flags | {"hold_max_tilt_deg"}:
        assert feedback[name] == diagnostics[name]
    assert set(feedback["terminal_block_footprints"]) == set(BLOCK_NAMES)
    for name, row in feedback["terminal_block_footprints"].items():
        assert set(row) == {"left", "right", "valid", "beam_xy_bounds_m"}
        assert row == diagnostics["terminal"]["block_footprints"][name]
    encoded = json.dumps(feedback)
    for private in ("theta", "hidden_mass_audit_kg", "nonstack_feasibility_witness",
                    "load_com", "gravity_line_arm"):
        assert private not in encoded


@pytest.mark.parametrize("track", ("a", "b"))
def test_t10_tracks_are_registered(track: str) -> None:
    """Both public tracks instantiate the registered implementation."""
    name = f"t10_hidden_mass_balance_beam_{track}"
    task = make_task(name)
    assert type(task) is TASKS[name]
    assert task.measurement_track == track.upper()
    assert task.extra_api_doc == (_MASS_MEASUREMENT_DOC if track == "a" else "")
    if track == "a":
        assert task.task_description.endswith(task.extra_api_doc)


@pytest.mark.parametrize("permutation", range(6))
def test_t10_nominal_witness_is_physically_reachable(permutation: int) -> None:
    """The oracle must execute its balanced geometric witness through RobotAPI."""
    task = make_task("t10_hidden_mass_balance_beam_b")
    theta = task.param_space.nominal()
    theta["mass_permutation"] = permutation
    witness = nonstack_feasibility_witness(theta)
    masses = masses_by_block(theta)
    assert sum(masses[name] * witness[name][0] for name in BLOCK_NAMES) == pytest.approx(0)
    for x, y in witness.values():
        assert ZONE_X[0] <= abs(x) - BLOCK_HALF_SIZE
        assert abs(x) + BLOCK_HALF_SIZE <= ZONE_X[1]
        assert abs(y) + BLOCK_HALF_SIZE <= ZONE_HALF_Y

    episode = task.make_episode(theta, seed=0)
    trace = run_policy(task.oracle_policy(theta), episode, theta, seed=0)
    diagnostics = episode._hidden_mass_balance_diagnostics
    print(json.dumps({"permutation": permutation, "success": trace.success,
                      "error": trace.error, "sim_time_s": trace.sim_time,
                      "diagnostics": diagnostics, "public_feedback": trace.public_feedback}, sort_keys=True))
    _assert_public_feedback(trace.public_feedback, diagnostics)
    assert trace.error is None
    assert trace.success, diagnostics


def test_t10_table_and_gripper_support_are_rejected() -> None:
    """Actual table and held-object contacts cannot satisfy beam support."""
    task = make_task("t10_hidden_mass_balance_beam_b")
    theta = task.param_space.nominal()
    episode = task.make_episode(theta, seed=0)
    assert not hasattr(episode, "extra_primitives")
    table_state = _step_state(episode.model, episode.data)
    assert table_state["external_block_support"]
    assert not any(table_state["block_supported_to_beam"].values())
    result = episode.evaluate()
    assert not result.success
    _assert_public_feedback(result.public_feedback, episode._hidden_mass_balance_diagnostics)

    api = episode.api
    pos = api.get_object_pose("block_a")["pos"]
    api.open_gripper()
    api.move_to([pos[0], pos[1], pos[2] + 0.10])
    api.move_to(pos)
    api.close_gripper()
    api.move_to([pos[0], pos[1], pos[2] + 0.15])
    assert api.get_object_pose("block_a")["pos"][2] > pos[2] + 0.10
    # The actual held cube has robot contact while clear of the table and beam.
    block_id = episode.model.body("block_a").id
    touching_bodies = {
        int(episode.model.geom_bodyid[c.geom2 if episode.model.geom_bodyid[c.geom1] == block_id else c.geom1])
        for c in episode.data.contact
        if block_id in (episode.model.geom_bodyid[c.geom1], episode.model.geom_bodyid[c.geom2])
    }
    assert touching_bodies
    assert touching_bodies <= {
        episode.model.body("gripper_link1").id,
        episode.model.body("gripper_link2").id,
    }
    held_state = _step_state(episode.model, episode.data)
    assert held_state["external_block_support"]
    assert not held_state["block_supported_to_beam"]["block_a"]
    result = episode.evaluate()
    assert not result.success
    _assert_public_feedback(result.public_feedback, episode._hidden_mass_balance_diagnostics)


def test_t10_mass_macro_returns_nan_without_physical_grasp() -> None:
    """A disabled gripper cannot reveal mass even when the full macro runs."""
    task = make_task("t10_hidden_mass_balance_beam_a")
    episode = task.make_episode(task.param_space.nominal(), seed=0)
    assert episode.extra_primitives == ("get_mass",)
    assert episode._inner.get_mass("block_a") is None
    # Zero motor gain creates a real failed grasp; no gate state is fabricated.
    episode.model.actuator_gainprm[6:8, 0] = 0.0
    measured = episode.api.get_mass("block_a")
    assert math.isnan(measured)
    assert not episode._physcap_mass_reveal.state("block_a")["revealed"]
    assert episode.api.measurements[-1]["return_status"] == "gate_not_passed"
