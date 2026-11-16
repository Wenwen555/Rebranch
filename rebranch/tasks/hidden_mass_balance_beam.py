"""KinDER-derived hidden-mass balance beam adapted to the local PiPER-X scene."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from rebranch.sim.backend import EpisodeResult, TaskBackend
from rebranch.sim.mujoco_episode import MujocoEpisode
from rebranch.sim.params import Continuous, Discrete, ParamSpace
from rebranch.sim.kinder_scene import (
    BEAM_CENTER,
    BEAM_CLEARANCE,
    BEAM_HALF_THICKNESS,
    BLOCK_HALF_SIZE,
    BLOCK_NAMES,
    HINGE_STIFFNESS_NM_RAD,
    HINGE_FRICTIONLOSS_NM,
    MASS_RATIO_BOUNDS,
    PAIR_GAP,
    PERMUTATIONS,
    PIVOT_HEIGHT,
    SUPPORT_Z,
    ZONE_HALF_Y,
    ZONE_X,
    build_hidden_mass_balance_model,
    masses_by_block,
    nonstack_feasibility_witness,
)
from rebranch.tasks._common import (
    DEFAULT_OBS_API_MAP,
    DEFAULT_OBS_SCALE,
    body_tilt_rad,
)

HOLD_S = 0.5
MAX_TILT_DEG = 5.0
WITHDRAW_M = 0.20
BEAM_BODY = "balance_support_beam"

_TASK_DESCRIPTION = (
    f"Balance three visually identical {2 * BLOCK_HALF_SIZE * 1000:.0f} mm cubes on the real "
    "hinged beam. Each cube has "
    "an unknown continuous mass; do not infer mass from the block name, start position, or "
    "appearance. The fixed support 'balance_support' is attached at public world pose "
    f"[{BEAM_CENTER[0]:.2f}, {BEAM_CENTER[1]:.2f}, {SUPPORT_Z:.3f}] m. Public objects are "
    "'block_a', 'block_b', and "
    "'block_c'; the world-frame beam pose is api.get_object_pose('beam'), and the beam-attached "
    "2D placement regions are "
    "api.get_object_pose('zone_left') and api.get_object_pose('zone_right'). For final loading, "
    "keep every cube footprint inside either marked region. Temporary single-block probes may "
    "use the supported beam outside these marked regions and must be retrieved before final "
    "loading. In beam coordinates, the final x ranges are "
    f"[{-ZONE_X[1]:.3f}, {-ZONE_X[0]:.3f}] m and "
    f"[{ZONE_X[0]:.3f}, {ZONE_X[1]:.3f}] m, with y range "
    f"[-{ZONE_HALF_Y:.3f}, {ZONE_HALF_Y:.3f}] m. Legal stacks are allowed. "
    f"The beam runs along world Y; its public spring is {HINGE_STIFFNESS_NM_RAD:.3f} "
    f"N m/rad and hinge friction loss is {HINGE_FRICTIONLOSS_NM:.3f} N m. "
    "After placing all cubes, "
    "release the gripper, withdraw at least 0.20 m from the beam, and keep the beam within 5 "
    "degrees of level for a measured 0.50 s hold. The evaluator checks real block support "
    "contact chains to the beam and rejects support from the robot or table. No instantaneous "
    "velocity threshold is used."
)

_MASS_MEASUREMENT_DOC = (
    "\nShared Track A measurement primitive for every method:\n"
    "- api.get_mass(name: str) -> float, mass in kg; NaN means the physical reveal gate "
    "was not satisfied.\n"
    "- One call runs the same PhysCaP simulator macro: approach 0.10 m above the named "
    "object, grasp, lift 0.15 m, hold 0.6 s, read the gated simulator mass, lower to "
    "5 mm above the start, hold 0.3 s, release, retreat, and return home.\n"
    "- The macro counts as one API call under the common 200-call episode cap. Its actual "
    "motion and hold time is charged to episode simulation time; the returned value is "
    "available only after the original contact/lift-duration gate passes.\n"
)


def _policy_source(layout: list[tuple[str, str, float, float, float]]) -> str:
    """Build a transfer program from public beam-attached region poses."""
    rows = "\n".join(
        f"    _place(api, {name!r}, {zone!r}, {dx:.8f}, {dy:.8f}, {stack:.8f})"
        for name, zone, dx, dy, stack in layout
    )
    return (
        "def run(api):\n"
        "    start = api.get_ee_pose()\n"
        "    def _world(site, dx, dy, dz):\n"
        "        p = site['pos']\n"
        "        q = site['quat']\n"
        "        w, x, y, z = q\n"
        "        rx = (1-2*(y*y+z*z))*dx + 2*(x*y-z*w)*dy + 2*(x*z+y*w)*dz\n"
        "        ry = 2*(x*y+z*w)*dx + (1-2*(x*x+z*z))*dy + 2*(y*z-x*w)*dz\n"
        "        rz = 2*(x*z-y*w)*dx + 2*(y*z+x*w)*dy + (1-2*(x*x+y*y))*dz\n"
        "        return [p[0]+rx, p[1]+ry, p[2]+rz]\n"
        "    def _place(api, name, zone_name, dx, dy, stack):\n"
        "        obj = api.get_object_pose(name)['pos']\n"
        "        api.open_gripper(0.068)\n"
        "        api.move_to([obj[0], obj[1], obj[2]+0.08], speed=0.05)\n"
        "        api.move_to(obj, speed=0.03)\n"
        "        api.close_gripper(force=10.0)\n"
        "        api.move_to([obj[0], obj[1], obj[2]+0.09], speed=0.04)\n"
        "        held = api.get_object_pose(name)['pos']\n"
        "        tcp = api.get_ee_pose()['pos']\n"
        "        grasp_offset = [held[i]-tcp[i] for i in range(3)]\n"
        "        site = api.get_object_pose(zone_name)\n"
        f"        target_z = {BLOCK_HALF_SIZE - 0.0003:.7f}+stack\n"
        "        target = _world(site, dx, dy, target_z)\n"
        "        tcp_target = [target[i]-grasp_offset[i] for i in range(3)]\n"
        "        api.move_to([tcp_target[0], tcp_target[1], tcp_target[2]+0.07], speed=0.04)\n"
        "        held = api.get_object_pose(name)['pos']\n"
        "        tcp = api.get_ee_pose()['pos']\n"
        "        grasp_offset = [held[i]-tcp[i] for i in range(3)]\n"
        "        tcp_target = [target[i]-grasp_offset[i] for i in range(3)]\n"
        "        api.move_to(tcp_target, quat=start['quat'], speed=0.02)\n"
        "        for _ in range(2):\n"
        "            actual = api.get_object_pose(name)['pos']\n"
        "            current_site = api.get_object_pose(zone_name)\n"
        "            desired = _world(current_site, dx, dy, target_z)\n"
        "            tcp = api.get_ee_pose()['pos']\n"
        "            tcp_target = [tcp[i]+desired[i]-actual[i] for i in range(3)]\n"
        "            api.move_to(tcp_target, quat=start['quat'], speed=0.02)\n"
        "        api.open_gripper(0.068)\n"
        "        tcp = api.get_ee_pose()['pos']\n"
        "        api.move_to([tcp[0], tcp[1], tcp[2]+0.07], quat=start['quat'], speed=0.04)\n"
        f"{rows}\n"
        "    beam = api.get_object_pose('beam')['pos']\n"
        "    api.move_to([beam[0], beam[1], 0.35], quat=start['quat'], speed=0.04)\n"
    )


def _step_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Read support contacts and beam-frame geometry for terminal scoring/auditing."""
    beam_id = model.body(BEAM_BODY).id
    beam_pos = data.xpos[beam_id].copy()
    beam_rotation = data.xmat[beam_id].reshape(3, 3).copy()
    block_ids = {name: model.body(name).id for name in BLOCK_NAMES}
    beam_geom_id = model.geom("balance_support_beam_geom").id
    block_geom_ids = {name: model.geom(f"{name}_geom").id for name in BLOCK_NAMES}
    block_id_set = set(block_ids.values())
    beam_geom_body = int(model.geom_bodyid[beam_geom_id])
    graph: dict[int, set[int]] = {body_id: set() for body_id in block_id_set}
    graph[beam_geom_body] = set()
    external_block_support = False
    beam_external_support = False

    contact_force = np.zeros(6, dtype=float)  # Reused normal-force scratch buffer (N).
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        if float(contact_force[0]) <= 1e-6:
            continue
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        body_a = int(model.geom_bodyid[geom_a])
        body_b = int(model.geom_bodyid[geom_b])
        if body_a in block_id_set:
            if body_b in block_id_set:
                graph[body_a].add(body_b)
            elif body_b == beam_geom_body:
                graph[body_a].add(beam_geom_body)
                graph[beam_geom_body].add(body_a)
            else:
                external_block_support = True
        if body_b in block_id_set:
            if body_a in block_id_set:
                graph[body_b].add(body_a)
            elif body_a == beam_geom_body:
                graph[body_b].add(beam_geom_body)
                graph[beam_geom_body].add(body_b)
            else:
                external_block_support = True
        if ((body_a == beam_geom_body and body_b not in block_id_set
             and body_b != model.body("balance_support").id)
                or (body_b == beam_geom_body and body_a not in block_id_set
                    and body_a != model.body("balance_support").id)):
            beam_external_support = True

    connected = {beam_geom_body}
    frontier = [beam_geom_body]
    while frontier:
        current = frontier.pop()
        for neighbor in graph[current]:
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)
    supported = {name: body_id in connected for name, body_id in block_ids.items()}

    block_footprints: dict[str, dict[str, Any]] = {}
    mass_sum = 0.0
    com_world = np.zeros(3, dtype=float)
    for name, body_id in block_ids.items():
        geom_id = block_geom_ids[name]
        geom_pos = data.geom_xpos[geom_id]
        geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
        half = model.geom_size[geom_id, :3]
        corners = np.array([
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ])
        beam_corners = (geom_pos + corners @ geom_rotation.T - beam_pos) @ beam_rotation
        x_min, y_min = np.min(beam_corners[:, :2], axis=0)
        x_max, y_max = np.max(beam_corners[:, :2], axis=0)
        left = x_min >= -ZONE_X[1] and x_max <= -ZONE_X[0]
        right = x_min >= ZONE_X[0] and x_max <= ZONE_X[1]
        within_width = y_min >= -ZONE_HALF_Y and y_max <= ZONE_HALF_Y
        footprint = (left or right) and within_width
        block_footprints[name] = {
            "left": bool(left and within_width),
            "right": bool(right and within_width),
            "valid": bool(footprint),
            "beam_xy_bounds_m": [float(x_min), float(x_max), float(y_min), float(y_max)],
        }
        mass = float(model.body_mass[body_id])
        mass_sum += mass
        com_world += mass * data.xipos[body_id]

    com_world /= mass_sum
    com_beam = beam_rotation.T @ (com_world - beam_pos)
    pivot_rotation = data.xmat[model.body("balance_support").id].reshape(3, 3)
    relative_rotation = pivot_rotation.T @ beam_rotation
    beam_angle = math.atan2(float(relative_rotation[0, 2]), float(relative_rotation[2, 2]))
    effective_com_x = float(com_beam[0] * math.cos(beam_angle)
                            + com_beam[2] * math.sin(beam_angle))
    return {
        "beam_pose_world": {
            "pos": beam_pos.astype(float).tolist(),
            "quat": data.xquat[beam_id].astype(float).tolist(),
        },
        "beam_tilt_deg": math.degrees(body_tilt_rad(data, beam_id)),
        "beam_pitch_deg": math.degrees(abs(beam_angle)),
        "block_footprints": block_footprints,
        "block_supported_to_beam": supported,
        "external_block_support": external_block_support,
        "beam_external_support": beam_external_support,
        "load_com_world_m": com_world.astype(float).tolist(),
        "load_com_beam_m": com_beam.astype(float).tolist(),
        "gravity_line_arm_beam_m": effective_com_x,
    }


class HiddenMassBalanceBeamTask(TaskBackend):
    """Shared fixed-support, native-hinge task; subclasses select measurement track."""

    measurement_track = "B"
    name = "t10_hidden_mass_balance_beam_b"
    requires_branching = False
    task_description = _TASK_DESCRIPTION

    def __init__(self) -> None:
        self.param_space = ParamSpace([
            Continuous("pair_mass_a_kg", 0.04, 0.075, nominal=0.055),
            Continuous("pair_mass_b_kg", 0.04, 0.075, nominal=0.055),
            Continuous("mass_ratio", *MASS_RATIO_BOUNDS, nominal=1.2),
            Discrete("mass_permutation", tuple(range(6)), nominal=0),
        ])
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = _policy_source([
            ("block_a", "zone_left", 0.0, 0.0, 0.0),
            ("block_b", "zone_right", 0.0, 0.0, 0.0),
            ("block_c", "zone_left", 0.0, 0.0, 2 * BLOCK_HALF_SIZE),
        ])
        if self.measurement_track == "A":
            self.task_description = _TASK_DESCRIPTION + _MASS_MEASUREMENT_DOC
        else:
            self.task_description = _TASK_DESCRIPTION + (
                "\nTrack B has no mass getter. Use public motion and beam-pose feedback only.\n"
            )

    def make_episode(self, theta: dict[str, Any], seed: int = 0):
        model = build_hidden_mass_balance_model(theta)
        masses = masses_by_block(theta)
        witness = nonstack_feasibility_witness(theta)
        beam_top = SUPPORT_Z + PIVOT_HEIGHT + BEAM_CLEARANCE + BEAM_HALF_THICKNESS

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            n_hold = int(round(HOLD_S / ep.dt))
            # Only conjunctions, a maximum and the final sample are needed; do not
            # retain a complete dictionary for every 500 Hz hold sample.
            all_footprints = all_supported = stable = released = withdrawn = True
            max_tilt = 0.0
            for _ in range(n_hold):
                ep.step_idle(1)
                final = _step_state(ep.model, ep.data)
                final["ee_pose_world"] = ep.get_ee_pose()
                final["released"] = not bool(ep.get_gripper_state()["grasped"])
                final["withdrawn"] = (
                    math.dist(final["ee_pose_world"]["pos"], final["beam_pose_world"]["pos"])
                    >= WITHDRAW_M
                )
                all_footprints &= all(row["valid"] for row in final["block_footprints"].values())
                all_supported &= (
                    all(final["block_supported_to_beam"].values())
                    and not final["external_block_support"]
                    and not final["beam_external_support"]
                )
                stable &= final["beam_tilt_deg"] <= MAX_TILT_DEG
                released &= final["released"]
                withdrawn &= final["withdrawn"]
                max_tilt = max(max_tilt, final["beam_tilt_deg"])
            success = all((all_footprints, all_supported, stable, released, withdrawn))
            quality = (
                0.24 * sum(final["block_footprints"][name]["valid"] for name in BLOCK_NAMES) / 3
                + 0.24 * sum(final["block_supported_to_beam"].values()) / 3
                + 0.24 * max(0.0, 1.0 - final["beam_tilt_deg"] / MAX_TILT_DEG)
                + 0.14 * float(released)
                + 0.14 * float(withdrawn)
            )
            score = quality if success else -max(0.01, 1.0 - quality)
            ep._hidden_mass_balance_diagnostics = {
                "track": self.measurement_track,
                "hold_steps": n_hold,
                "hold_sim_time_s": n_hold * ep.dt,
                "all_footprints_in_2d_regions": all_footprints,
                "all_support_chains_reach_beam": all_supported,
                "beam_stable_throughout_hold": stable,
                "released_throughout_hold": released,
                "withdrawn_throughout_hold": withdrawn,
                "hold_max_tilt_deg": max_tilt,
                "terminal": final,
                "hidden_mass_audit_kg": masses,
                "nonstack_feasibility_witness_beam_xy_m": witness,
                "engineering_default_beam_top_z_m": beam_top,
            }
            return EpisodeResult(
                success=success,
                score=max(0.01, score) if success else min(-0.01, score),
                failure=None if success else "balance_goal_not_satisfied",
            )

        episode = MujocoEpisode(
            theta=theta,
            seed=seed,
            model=model,
            evaluate_fn=evaluate,
        )
        if self.measurement_track == "A":
            from rebranch.baselines.physcap_appendix import with_physcap_mass_macro

            return with_physcap_mass_macro(episode)
        return episode

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        witness = nonstack_feasibility_witness(theta)
        permutation = PERMUTATIONS[int(theta["mass_permutation"])]
        name_by_role = {
            role: BLOCK_NAMES[block_index]
            for block_index, role in enumerate(permutation)
        }
        layout = [
            (
                name,
                "zone_left" if x < 0.0 else "zone_right",
                x + (ZONE_X[0] + ZONE_X[1]) / 2
                if x < 0.0 else x - (ZONE_X[0] + ZONE_X[1]) / 2,
                y,
                0.0,
            )
            for name in (name_by_role[0], name_by_role[2], name_by_role[1])
            for x, y in (witness[name],)
        ]
        return _policy_source(layout)

    def conservative_policy(self) -> str:
        return self.draft_policy

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return "continuous_unknown_mass"


class HiddenMassBalanceBeamATask(HiddenMassBalanceBeamTask):
    measurement_track = "A"
    name = "t10_hidden_mass_balance_beam_a"


class HiddenMassBalanceBeamBTask(HiddenMassBalanceBeamTask):
    measurement_track = "B"
