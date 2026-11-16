"""Pilot task: place two unknown-mass cans on a free tray over a raised support edge."""
from __future__ import annotations

import math
import itertools
from typing import Any

import mujoco
import numpy as np

from rebranch.sim.backend import Episode, EpisodeResult, TaskBackend
from rebranch.sim.mujoco_episode import MujocoEpisode
from rebranch.sim.params import Continuous, ParamSpace
from rebranch.tasks._common import DEFAULT_OBS_API_MAP, DEFAULT_OBS_SCALE, body_tilt_rad

TRAY_CENTER = (0.37, 0.0)
TRAY_HALF = (0.15, 0.12, 0.006)
TRAY_MASS_KG = 0.18
SUPPORT_CENTER_X = 0.315
SUPPORT_HALF_X = 0.095
SUPPORT_HALF_Y = 0.26
STEP_HEIGHT = 0.075
CAN_RADIUS = 0.022
CAN_HALF_HEIGHT = 0.035
CAN_MASS_BOUNDS = (0.04, 0.30)
SLOT_MARK_RADIUS = 0.034
SLOT_CENTER_TOL = 0.010
STABILITY_HOLD_S = 1.0
MAX_TRAY_LINEAR_SPEED = 0.010
MAX_TRAY_ANGULAR_SPEED = 0.050
MAX_CAN_LINEAR_SPEED = 0.010
MAX_CAN_ANGULAR_SPEED = 0.050
PICKUPS = {"can_a": (0.30, -0.19), "can_b": (0.30, 0.19)}
SLOTS = {"inner": (0.32, 0.0), "outer": (0.47, 0.0)}
TARGET_X_LOCAL = {key: value[0] - TRAY_CENTER[0] for key, value in SLOTS.items()}
ROBOT_WORKSPACE = ((0.22, 0.48), (-0.22, 0.22))


def enumerate_static_plans(mass_a_kg: float, mass_b_kg: float) -> list[dict[str, Any]]:
    """Enumerate two assignments and both orders using support-polygon statics.

    This is a feasibility audit/strategy helper; task success never consults it.
    It checks each released prefix, target reachability, target separation and
    the combined horizontal COM projection over the raised support footprint.
    """
    masses = {"can_a": float(mass_a_kg), "can_b": float(mass_b_kg)}
    support_x = (SUPPORT_CENTER_X - SUPPORT_HALF_X, SUPPORT_CENTER_X + SUPPORT_HALF_X)
    support_y = (-SUPPORT_HALF_Y, SUPPORT_HALF_Y)
    target_values = list(SLOTS.values())
    target_separation = math.dist(target_values[0], target_values[1])
    if target_separation <= 2.0 * CAN_RADIUS:
        return []
    if any(
        not (ROBOT_WORKSPACE[0][0] <= x <= ROBOT_WORKSPACE[0][1]
             and ROBOT_WORKSPACE[1][0] <= y <= ROBOT_WORKSPACE[1][1])
        for x, y in target_values
    ):
        return []
    # A square cover of the public 10 mm radial tolerance disk is checked.
    offsets = ((-SLOT_CENTER_TOL, -SLOT_CENTER_TOL),
               (-SLOT_CENTER_TOL, SLOT_CENTER_TOL),
               (SLOT_CENTER_TOL, -SLOT_CENTER_TOL),
               (SLOT_CENTER_TOL, SLOT_CENTER_TOL))

    plans: list[dict[str, Any]] = []
    for inner_item in ("can_a", "can_b"):
        outer_item = "can_b" if inner_item == "can_a" else "can_a"
        assignment = {inner_item: "inner", outer_item: "outer"}
        for first in ("can_a", "can_b"):
            order = (first, "can_b" if first == "can_a" else "can_a")
            placed: list[str] = []
            margins: list[float] = []
            feasible = True
            for item in order:
                placed.append(item)
                prefix_margin = float("inf")
                total_mass = TRAY_MASS_KG + sum(masses[name] for name in placed)
                for location_combo in itertools.product(offsets, repeat=len(placed)):
                    x_moment = TRAY_MASS_KG * TRAY_CENTER[0]
                    y_moment = 0.0
                    for name, (ox, oy) in zip(placed, location_combo):
                        sx, sy = SLOTS[assignment[name]]
                        x_moment += masses[name] * (sx + ox)
                        y_moment += masses[name] * (sy + oy)
                    com_x = x_moment / total_mass
                    com_y = y_moment / total_mass
                    mx = min(com_x - support_x[0], support_x[1] - com_x)
                    my = min(com_y - support_y[0], support_y[1] - com_y)
                    prefix_margin = min(prefix_margin, mx, my)
                    if mx < 0.0 or my < 0.0:
                        feasible = False
                margins.append(prefix_margin)
                if not feasible:
                    break
            if feasible:
                plans.append({"assignment": assignment, "order": order,
                              "min_static_margin_m": min(margins)})
    return sorted(plans, key=lambda plan: plan["min_static_margin_m"], reverse=True)


def episode_diagnostics(ep: MujocoEpisode) -> dict[str, Any]:
    """Report observed poses, terminal motion/contact and static-plan evidence."""
    model, data = ep.model, ep.data
    tray_bid = model.body("tray").id
    can_bids = {name: model.body(name).id for name in ("can_a", "can_b")}
    task_bids = {tray_bid, *can_bids.values()}
    tray_gid = model.geom("tray_geom").id
    support_gid = model.geom("support_step").id
    floor_gid = model.geom("catch_floor").id
    robot_task_contacts = []
    tray_support = False
    tray_floor = False
    can_floor = {"can_a": False, "can_b": False}
    can_tray = {"can_a": False, "can_b": False}
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        pair = {g1, g2}
        if tray_gid in pair and support_gid in pair:
            tray_support = True
        if tray_gid in pair and floor_gid in pair:
            tray_floor = True
        for name in can_bids:
            can_gid = model.geom(f"{name}_geom").id
            if can_gid in pair and tray_gid in pair:
                can_tray[name] = True
            if can_gid in pair and floor_gid in pair:
                can_floor[name] = True
        if ((b1 in task_bids and b2 not in task_bids and b2 != 0)
                or (b2 in task_bids and b1 not in task_bids and b1 != 0)):
            robot_task_contacts.append((g1, g2))

    tray_jid = int(model.joint("tray_free").id)
    tray_dof = int(model.jnt_dofadr[tray_jid])
    tray_speed = data.qvel[tray_dof:tray_dof + 3]
    tray_omega = data.qvel[tray_dof + 3:tray_dof + 6]
    tray_R = data.xmat[tray_bid].reshape(3, 3)
    can_speeds = {}
    can_z_axes = {}
    for name in can_bids:
        jid = int(model.joint(f"{name}_free").id)
        dof = int(model.jnt_dofadr[jid])
        can_z_axes[name] = data.xmat[can_bids[name]].reshape(3, 3)[:, 2].astype(float).tolist()
        can_speeds[name] = {
            "linear_m_s": float(np.linalg.norm(data.qvel[dof:dof + 3])),
            "angular_rad_s": float(np.linalg.norm(data.qvel[dof + 3:dof + 6])),
        }
    masses = {name: float(model.body_mass[bid]) for name, bid in can_bids.items()}
    return {
        "tray_pose_world": ep.get_object_pose("tray"),
        "ee_pose_world": ep.get_ee_pose(),
        "tray_rotation_world": tray_R.astype(float).reshape(-1).tolist(),
        "tray_z_axis_world": tray_R[:, 2].astype(float).tolist(),
        "can_poses_world": {name: ep.get_object_pose(name) for name in can_bids},
        "can_z_axes_world": can_z_axes,
        "tray_tilt_deg": math.degrees(body_tilt_rad(data, tray_bid)),
        "tray_linear_speed_m_s": float(np.linalg.norm(tray_speed)),
        "tray_angular_speed_rad_s": float(np.linalg.norm(tray_omega)),
        "can_speeds": can_speeds,
        "tray_support_contact": tray_support,
        "can_tray_contact": can_tray,
        "tray_catch_floor_contact": tray_floor,
        "can_catch_floor_contact": can_floor,
        "robot_task_contact_pairs": robot_task_contacts,
        "gripper": ep.get_gripper_state(),
        "sim_time_s": float(ep.sim_time),
        # Written only to local audit files; never included in API observations.
        "physics_mass_audit_kg": masses,
    }


class T9SupportedTrayTask(TaskBackend):
    """Move two identical outside-shape cans onto the marked positions of a free tray."""

    name = "t9_supported_tray"
    probe_config = None  # The pilot's fixed-probe baseline explicitly measures both named cans.
    requires_branching = True
    task_description = (
        "Place one can at each of the named marks 'slot_inner' and 'slot_outer' on the free "
        "rigid tray 'tray', then release the gripper and retreat. The tray rests on the raised "
        "support step 'support_step' (x=0.220..0.410 m, y=-0.260..0.260 m) at its inner end "
        "and overhangs its outer edge; the lower "
        "catch floor is not a second tray support. Both cans have the same visible cylindrical "
        "shape. Their masses are unknown and may differ. The tray, support edge, pickup positions "
        "'can_a'/'can_b', and two slot marks are publicly named. Each mark is a 68 mm diameter "
        "circular target; each can has a 44 mm diameter base. The can base center must be within "
        "10 mm radial distance of its mark, leaving at least 2 mm footprint margin. The publicly "
        "named fixed layout at x=0.314 m and x=0.464 m uses a 6 mm inward offset and is available "
        "to every strategy. After release "
        "and retreat, maintain this placement for a measured 1.0 s window. Throughout that window "
        "the tray must contact the raised support, both cans must contact the tray, and neither "
        "tray nor can may touch the lower catch floor or robot. Tray tilt must remain <=5 deg, "
        "tray height within 10 mm of supported height, tray/can linear speed <=0.010 m/s and "
        "angular speed <=0.050 rad/s. The robot must finish at least 0.200 m from the tray center."
    )

    def __init__(self) -> None:
        lo, hi = CAN_MASS_BOUNDS
        self.param_space = ParamSpace([
            Continuous("mass_a_kg", lo, hi, nominal=0.17),
            Continuous("mass_b_kg", lo, hi, nominal=0.17),
        ])
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = self.conservative_policy()

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        mass_a = float(theta["mass_a_kg"])
        mass_b = float(theta["mass_b_kg"])
        if not (CAN_MASS_BOUNDS[0] <= mass_a <= CAN_MASS_BOUNDS[1]
                and CAN_MASS_BOUNDS[0] <= mass_b <= CAN_MASS_BOUNDS[1]):
            raise ValueError("can masses are outside the declared physical domain")
        tray_x, tray_y = TRAY_CENTER
        tray_z = STEP_HEIGHT + TRAY_HALF[2]
        support_pos = [SUPPORT_CENTER_X, 0.0, STEP_HEIGHT / 2.0]
        support_size = [SUPPORT_HALF_X, SUPPORT_HALF_Y, STEP_HEIGHT / 2.0]
        pickup_z = STEP_HEIGHT + CAN_HALF_HEIGHT

        def mutate(spec: mujoco.MjSpec) -> None:
            spec.worldbody.add_geom(
                name="support_step", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=support_pos, size=support_size,
                friction=[0.8, 0.005, 0.0001], rgba=[0.48, 0.48, 0.50, 1.0],
            )
            spec.worldbody.add_geom(
                name="catch_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0.35, 0.0, 0.0], size=[1.0, 1.0, 0.1],
                friction=[0.8, 0.005, 0.0001], rgba=[0.72, 0.73, 0.74, 1.0],
            )
            tray = spec.worldbody.add_body(name="tray", pos=[tray_x, tray_y, tray_z])
            tray.add_joint(name="tray_free", type=mujoco.mjtJoint.mjJNT_FREE)
            tray.mass = TRAY_MASS_KG
            tray.ipos = [0.0, 0.0, 0.0]
            hx, hy, hz = TRAY_HALF
            tray.inertia = [
                TRAY_MASS_KG * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12.0,
                TRAY_MASS_KG * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12.0,
                TRAY_MASS_KG * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12.0,
            ]
            tray.add_geom(name="tray_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=list(TRAY_HALF), friction=[0.8, 0.005, 0.0001],
                          rgba=[0.24, 0.45, 0.68, 1.0])
            tray.add_site(name="slot_inner", pos=[SLOTS["inner"][0] - tray_x, 0.0, hz])
            tray.add_site(name="slot_outer", pos=[SLOTS["outer"][0] - tray_x, 0.0, hz])
            # Visible, non-colliding circular marks; center tolerance keeps the
            # entire 44 mm can base inside each 68 mm mark with 2 mm margin.
            for slot_name, (sx, sy) in SLOTS.items():
                tray.add_geom(
                    name=f"mark_{slot_name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[sx - tray_x, sy, hz + 0.00015],
                    size=[SLOT_MARK_RADIUS, 0.00015, 0.0],
                    contype=0, conaffinity=0,
                    rgba=[0.92, 0.78, 0.20, 1.0],
                )

            for name, mass in (("can_a", mass_a), ("can_b", mass_b)):
                px, py = PICKUPS[name]
                can = spec.worldbody.add_body(name=name, pos=[px, py, pickup_z])
                can.add_joint(name=f"{name}_free", type=mujoco.mjtJoint.mjJNT_FREE)
                can.mass = mass
                can.ipos = [0.0, 0.0, 0.0]
                ix = mass * (3 * CAN_RADIUS**2 + 4 * CAN_HALF_HEIGHT**2) / 12.0
                iz = 0.5 * mass * CAN_RADIUS**2
                can.inertia = [ix, ix, iz]
                can.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             size=[CAN_RADIUS, CAN_HALF_HEIGHT, 0.0],
                             friction=[0.85, 0.005, 0.0001],
                             rgba=[0.78, 0.58, 0.30, 1.0])

        step_log: list[dict[str, Any]] = []

        def record_physics_sample(model: mujoco.MjModel, data: mujoco.MjData,
                                  episode: MujocoEpisode) -> None:
            step = int(round(episode.sim_time / episode.dt))
            if step % 50:
                return
            touching = []
            can_geom_ids = {model.geom(f"{name}_geom").id for name in ("can_a", "can_b")}
            for ci in range(data.ncon):
                contact = data.contact[ci]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if g1 in can_geom_ids or g2 in can_geom_ids:
                    touching.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                     mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)))
            step_log.append({
                "sim_time_s": float(episode.sim_time),
                "ee_pos": data.site_xpos[episode.tcp_id].astype(float).tolist(),
                "can_a_pos": data.xpos[model.body("can_a").id].astype(float).tolist(),
                "can_b_pos": data.xpos[model.body("can_b").id].astype(float).tolist(),
                "gripper_width_m": float(episode._last_grip_width),
                "gripper_effort_n": float(episode._last_grip_effort),
                "grasped": bool(episode.get_gripper_state()["grasped"]),
                "joint_positions_rad": data.qpos[:6].astype(float).tolist(),
                "can_contacts": touching,
            })

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            n_hold = int(round(STABILITY_HOLD_S / ep.dt))
            hold_rows = []
            for _ in range(n_hold):
                ep.step_idle(1)
                hold_rows.append(episode_diagnostics(ep))
            diag = hold_rows[-1]
            ep._t9_diagnostics = {
                "terminal": diag, "hold_steps": n_hold, "hold_sim_time_s": n_hold * ep.dt,
                "hold_max": {
                    "tray_linear_speed_m_s": max(r["tray_linear_speed_m_s"] for r in hold_rows),
                    "tray_angular_speed_rad_s": max(r["tray_angular_speed_rad_s"] for r in hold_rows),
                    "can_linear_speed_m_s": max(v["linear_m_s"] for r in hold_rows for v in r["can_speeds"].values()),
                    "can_angular_speed_rad_s": max(v["angular_rad_s"] for r in hold_rows for v in r["can_speeds"].values()),
                },
                "hold_all_support": all(r["tray_support_contact"] for r in hold_rows),
                "hold_all_can_tray": all(all(r["can_tray_contact"].values()) for r in hold_rows),
                "hold_no_tray_floor": all(not r["tray_catch_floor_contact"] for r in hold_rows),
                "hold_no_can_floor": all(not any(r["can_catch_floor_contact"].values()) for r in hold_rows),
                "hold_no_floor": all(not r["tray_catch_floor_contact"] and not any(r["can_catch_floor_contact"].values()) for r in hold_rows),
                "hold_no_robot_contact": all(not r["robot_task_contact_pairs"] for r in hold_rows),
            }
            target_z = TRAY_HALF[2] + CAN_HALF_HEIGHT
            assignments = {
                "can_a_inner_can_b_outer": {"can_a": "inner", "can_b": "outer"},
                "can_b_inner_can_a_outer": {"can_b": "inner", "can_a": "outer"},
            }
            assignment_hold_valid = {name: True for name in assignments}
            error_max = {name: {item: {"radial_xy_m": 0.0, "vertical_m": 0.0,
                                       "min_upright_cos": 1.0}
                                for item in assignment}
                         for name, assignment in assignments.items()}
            board_stable_all = True
            released_all = True
            withdrawn_all = True
            robot_clear_all = True
            for row in hold_rows:
                board_pose = row["tray_pose_world"]["pos"]
                board_stable = (
                    float(row["tray_tilt_deg"]) <= 5.0
                    and abs(float(board_pose[2]) - tray_z) <= 0.010
                )
                board_stable_all = board_stable_all and board_stable
                released_all = released_all and not bool(row["gripper"]["grasped"])
                withdrawn_all = withdrawn_all and math.dist(row["ee_pose_world"]["pos"], board_pose) >= 0.200
                robot_clear_all = robot_clear_all and not row["robot_task_contact_pairs"]
                R = np.asarray(row["tray_rotation_world"], dtype=float).reshape(3, 3)
                tray_pos = np.asarray(board_pose, dtype=float)
                for assignment_name, assignment in assignments.items():
                    row_ok = board_stable
                    for item, target in assignment.items():
                        can_pos = np.asarray(row["can_poses_world"][item]["pos"], dtype=float)
                        rel = R.T @ (can_pos - tray_pos)
                        dx = float(rel[0]) - TARGET_X_LOCAL[target]
                        dy = float(rel[1])
                        dz = abs(float(rel[2]) - target_z)
                        upright_cos = float(np.dot(row["tray_z_axis_world"], row["can_z_axes_world"][item]))
                        err = error_max[assignment_name][item]
                        err["radial_xy_m"] = max(err["radial_xy_m"], math.hypot(dx, dy))
                        err["vertical_m"] = max(err["vertical_m"], dz)
                        err["min_upright_cos"] = min(err["min_upright_cos"], upright_cos)
                        row_ok = row_ok and (
                            math.hypot(dx, dy) <= SLOT_CENTER_TOL and dz <= 0.014
                            and upright_cos >= math.cos(math.radians(12.0))
                            and bool(row["can_tray_contact"][item])
                        )
                    assignment_hold_valid[assignment_name] = assignment_hold_valid[assignment_name] and row_ok
            target_occupancy = any(assignment_hold_valid.values())
            ep._t9_diagnostics["target_center_errors_by_assignment"] = error_max
            ep._t9_diagnostics["assignment_valid_for_full_hold"] = assignment_hold_valid
            ep._t9_diagnostics["hold_all_board_stable"] = board_stable_all
            hold = ep._t9_diagnostics
            settled = (
                hold["hold_max"]["tray_linear_speed_m_s"] <= MAX_TRAY_LINEAR_SPEED
                and hold["hold_max"]["tray_angular_speed_rad_s"] <= MAX_TRAY_ANGULAR_SPEED
                and hold["hold_max"]["can_linear_speed_m_s"] <= MAX_CAN_LINEAR_SPEED
                and hold["hold_max"]["can_angular_speed_rad_s"] <= MAX_CAN_ANGULAR_SPEED
                and hold["hold_all_support"] and hold["hold_all_can_tray"]
                and hold["hold_no_floor"] and hold["hold_no_robot_contact"]
            )
            supported = bool(ep._t9_diagnostics["hold_all_support"])
            no_fallen_can = bool(ep._t9_diagnostics["hold_no_can_floor"])
            no_tray_floor = bool(ep._t9_diagnostics["hold_no_tray_floor"])
            good = (
                ep.safety_violation is None and board_stable_all and target_occupancy
                and released_all and withdrawn_all and robot_clear_all and settled and supported
                and no_fallen_can and no_tray_floor
            )
            if good:
                return EpisodeResult(success=True, score=0.9)
            reasons = []
            if not board_stable_all: reasons.append("tray_unstable")
            if not target_occupancy: reasons.append("target_occupancy")
            if not released_all: reasons.append("gripper_not_released")
            if not withdrawn_all or not robot_clear_all: reasons.append("robot_not_clear")
            if not settled: reasons.append("not_settled")
            if not supported: reasons.append("tray_not_supported")
            if not no_fallen_can: reasons.append("can_fell_to_catch_floor")
            if not no_tray_floor: reasons.append("tray_caught_by_floor")
            if ep.safety_violation is not None: reasons.append(str(ep.safety_violation))
            return EpisodeResult(success=False, score=-0.5, failure="|".join(reasons))

        ep = MujocoEpisode(theta=theta, seed=seed, spec_mutator=mutate,
                           evaluate_fn=evaluate, step_callback=record_physics_sample,
                           settle_steps=100)
        ep._t9_step_log = step_log
        return ep

    def conservative_policy(self) -> str:
        return self._policy_program(
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            "    pick_place(api, \"can_a\", \"slot_inner\", -0.006)\n"
            "    pick_place(api, \"can_b\", \"slot_outer\", -0.006)\n"
            "    api.move_to([0.30, -0.20, 0.22], speed=PLACE_SPEED)\n"
        )

    @staticmethod
    def _policy_program(run_body: str, helper_blocks: str = "") -> str:
        """Attach the common observed-offset pick/lift/place control to each baseline."""
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "PLACE_SPEED = 0.025  # [0.01, 0.20]\n\n"
            "def grasp_and_lift(api, name):\n"
            "    p = api.get_object_pose(name)[\"pos\"]\n"
            "    api.open_gripper(0.067)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=PLACE_SPEED)\n"
            "    api.move_to(p, speed=PLACE_SPEED)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=PLACE_SPEED)\n"
            "    return p, api.get_gripper_state()[\"grasped\"]\n\n"
            "def place_held(api, name, target_x, target_y, target_center_z, slot=None, dx=0.0):\n"
            "    e = api.get_ee_pose()[\"pos\"]\n"
            "    o = api.get_object_pose(name)[\"pos\"]\n"
            "    tx = target_x + e[0] - o[0]\n"
            "    ty = target_y + e[1] - o[1]\n"
            "    approach_z = target_center_z + 0.020 + e[2] - o[2]\n"
            "    api.move_to([tx, ty, approach_z], speed=PLACE_SPEED)\n"
            "    if not api.get_gripper_state()[\"grasped\"]:\n"
            "        return False\n"
            "    if slot is not None:\n"
            "        s = api.get_object_pose(slot)[\"pos\"]\n"
            "        target_x = s[0] + dx\n"
            "        target_y = s[1]\n"
            "        target_center_z = s[2] + 0.035\n"
            "    e = api.get_ee_pose()[\"pos\"]\n"
            "    o = api.get_object_pose(name)[\"pos\"]\n"
            "    tx = target_x + e[0] - o[0]\n"
            "    ty = target_y + e[1] - o[1]\n"
            "    tz = target_center_z + 0.008 + e[2] - o[2]\n"
            "    api.move_to([tx, ty, tz], speed=PLACE_SPEED)\n"
            "    api.open_gripper(0.067)\n"
            "    api.move_to([tx, ty, tz + 0.10], speed=PLACE_SPEED)\n"
            "    return True\n\n"
            "def pick_place(api, name, slot, dx):\n"
            "    p, grasped = grasp_and_lift(api, name)\n"
            "    if not grasped:\n"
            "        return False\n"
            "    s = api.get_object_pose(slot)[\"pos\"]\n"
            "    return place_held(api, name, s[0] + dx, s[1], s[2] + 0.035, slot, dx)\n\n"
            + helper_blocks
            + run_body
        )

    def probe_policy(self) -> str:
        helper_blocks = (
            "def mass_probe(api, name):\n"
            "    p = api.get_object_pose(name)[\"pos\"]\n"
            "    api.open_gripper(0.040)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=PLACE_SPEED)\n"
            "    h = api.get_ee_pose()\n"
            "    api.move_compliant(h[\"pos\"], quat=h[\"quat\"], duration=0.20)\n"
            "    api.zero_ee_wrench()\n"
            "    base_sum = 0.0\n"
            "    for _ in range(3):\n"
            "        api.move_compliant(h[\"pos\"], quat=h[\"quat\"], duration=0.15)\n"
            "        base_sum += api.get_ee_wrench(\"world\")[\"force\"][2]\n"
            "    base_fz = base_sum / 3.0\n"
            "    api.open_gripper(0.067)\n"
            "    api.move_to(p, speed=PLACE_SPEED)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=PLACE_SPEED)\n"
            "    if not api.get_gripper_state()[\"grasped\"]:\n"
            "        return 0.0\n"
            "    load_sum = 0.0\n"
            "    for _ in range(3):\n"
            "        api.move_compliant(h[\"pos\"], quat=h[\"quat\"], duration=0.15)\n"
            "        load_sum += api.get_ee_wrench(\"world\")[\"force\"][2]\n"
            "    load_fz = load_sum / 3.0\n"
            "    mass_kg = abs(load_fz - base_fz) / 9.81\n"
            "    place_held(api, name, p[0], p[1], p[2])\n"
            "    return mass_kg\n\n"
        )
        run_body = (
            "def run(api):\n"
            "    mass_a = mass_probe(api, \"can_a\")\n"
            "    mass_b = mass_probe(api, \"can_b\")\n"
            "    plans = []\n"
            "    for inner in [\"can_a\", \"can_b\"]:\n"
            "        outer = \"can_b\" if inner == \"can_a\" else \"can_a\"\n"
            "        for first in [\"can_a\", \"can_b\"]:\n"
            "            first_mass = mass_a if first == \"can_a\" else mass_b\n"
            "            second_mass = mass_b if first == \"can_a\" else mass_a\n"
            "            inner_x = 0.314\n"
            "            outer_x = 0.464\n"
            "            first_x = inner_x if first == inner else outer_x\n"
            "            total_1 = 0.18 + first_mass\n"
            "            com_1 = (0.18 * 0.37 + first_mass * first_x) / total_1\n"
            "            total_2 = 0.18 + mass_a + mass_b\n"
            "            com_2 = (0.18 * 0.37 + mass_a * (inner_x if inner == \"can_a\" else outer_x) + mass_b * (inner_x if inner == \"can_b\" else outer_x)) / total_2\n"
            "            if 0.22 <= com_1 <= 0.41 and 0.22 <= com_2 <= 0.41:\n"
            "                margin = min(com_1 - 0.22, 0.41 - com_1, com_2 - 0.22, 0.41 - com_2)\n"
            "                plans.append((margin, inner, outer, first))\n"
            "    plans = sorted(plans, reverse=True)\n"
            "    if len(plans) == 0:\n"
            "        inner = \"can_a\" if mass_a >= mass_b else \"can_b\"\n"
            "        first = inner\n"
            "    else:\n"
            "        inner = plans[0][1]\n"
            "        first = plans[0][3]\n"
            "    second = \"can_b\" if first == \"can_a\" else \"can_a\"\n"
            "    first_slot = \"slot_inner\" if first == inner else \"slot_outer\"\n"
            "    second_slot = \"slot_inner\" if second == inner else \"slot_outer\"\n"
            "    pick_place(api, first, first_slot, -0.006)\n"
            "    pick_place(api, second, second_slot, -0.006)\n"
            "    api.move_to([0.30, -0.20, 0.22], speed=PLACE_SPEED)\n"
        )
        return self._policy_program(run_body, helper_blocks)

    def feedback_policy(self) -> str:
        run_body = (
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            "    pickup_a = api.get_object_pose(\"can_a\")[\"pos\"]\n"
            "    pickup_b = api.get_object_pose(\"can_b\")[\"pos\"]\n"
            "    pick_place(api, \"can_a\", \"slot_inner\", -0.006)\n"
            "    p, grasped = grasp_and_lift(api, \"can_b\")\n"
            "    if grasped:\n"
            "        warning = lower_trial_while_held(api, \"can_b\", \"slot_outer\", -0.006)\n"
            "        if warning and api.get_gripper_state()[\"grasped\"]:\n"
            "            place_held(api, \"can_b\", pickup_b[0], pickup_b[1], pickup_b[2])\n"
            "            move_can(api, \"can_a\", pickup_a[0], pickup_a[1], pickup_a[2])\n"
            "            pick_place(api, \"can_a\", \"slot_outer\", -0.006)\n"
            "            pick_place(api, \"can_b\", \"slot_inner\", -0.006)\n"
            "    api.move_to([0.30, -0.20, 0.22], speed=PLACE_SPEED)\n"
        )
        helper_blocks = (
            "def move_can(api, name, target_x, target_y, target_z):\n"
            "    p, grasped = grasp_and_lift(api, name)\n"
            "    if not grasped:\n"
            "        return False\n"
            "    return place_held(api, name, target_x, target_y, target_z)\n\n"
            "def lower_trial_while_held(api, name, slot, dx):\n"
            "    s = api.get_object_pose(slot)[\"pos\"]\n"
            "    e = api.get_ee_pose()[\"pos\"]\n"
            "    o = api.get_object_pose(name)[\"pos\"]\n"
            "    tx = s[0] + dx + e[0] - o[0]\n"
            "    ty = s[1] + e[1] - o[1]\n"
            "    center_z = s[2] + 0.035\n"
            "    api.move_to([tx, ty, center_z + 0.020 + e[2] - o[2]], speed=PLACE_SPEED)\n"
            "    for dz in [0.010, 0.004, 0.000, -0.002]:\n"
            "        s = api.get_object_pose(slot)[\"pos\"]\n"
            "        e = api.get_ee_pose()[\"pos\"]\n"
            "        o = api.get_object_pose(name)[\"pos\"]\n"
            "        tx = s[0] + dx + e[0] - o[0]\n"
            "        ty = s[1] + e[1] - o[1]\n"
            "        center_z = s[2] + 0.035\n"
            "        api.move_to([tx, ty, center_z + dz + e[2] - o[2]], speed=PLACE_SPEED)\n"
            "        e = api.get_ee_pose()\n"
            "        api.move_compliant(e[\"pos\"], quat=e[\"quat\"], duration=0.10)\n"
            "        q = api.get_object_pose(\"tray\")[\"quat\"]\n"
            "        if abs(q[1]) + abs(q[2]) > 0.020:\n"
            "            e = api.get_ee_pose()[\"pos\"]\n"
            "            api.move_to([e[0], e[1], e[2] + 0.08], speed=PLACE_SPEED)\n"
            "            return True\n"
            "    api.open_gripper(0.067)\n"
            "    e = api.get_ee_pose()[\"pos\"]\n"
            "    api.move_to([e[0], e[1], e[2] + 0.10], speed=PLACE_SPEED)\n"
            "    return False\n\n"
        )
        return self._policy_program(run_body, helper_blocks)

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        # The pilot does not use a privileged oracle; return the same non-theta baseline.
        return self.conservative_policy()

    def latent_mode(self, theta: dict[str, Any]) -> str:
        # Internal bookkeeping only; neither observations nor success use this label.
        return "a_heavier" if float(theta["mass_a_kg"]) > float(theta["mass_b_kg"]) else (
            "b_heavier" if float(theta["mass_b_kg"]) > float(theta["mass_a_kg"]) else "equal")
