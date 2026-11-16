"""T1: Eccentric CoM box pick-and-insert into narrow slot."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from rebranch.sim.backend import Episode, EpisodeResult, TaskBackend
from rebranch.sim.mujoco_episode import MujocoEpisode
from rebranch.sim.params import Continuous, Discrete, ParamSpace
from rebranch.tasks._common import (
    DEFAULT_OBS_API_MAP,
    DEFAULT_OBS_SCALE,
    TABLE_Z,
    add_table,
    body_tilt_rad,
)

_BOX_START = (0.33, -0.09, TABLE_Z + 0.020)
_BOX_HALF = (0.045, 0.021, 0.020)  # width 42 mm <= 55 mm
_SLOT_CENTER = (0.35, 0.08)
_SLOT_WALL_H = 0.030


class T1EccentricBoxTask(TaskBackend):
    """Pick an eccentric-CoM box and insert it flat into a narrow tabletop slot."""

    name = "t1_eccentric_box"
    probe_config = ('box', 'above', True)
    task_description = (
        "Pick the 90 x 42 x 40 mm box 'box' and place it flat in the slot 'slot'. "
        "Slot pose z is the tabletop floor; box half-height is 20 mm. "
        "Mass, longitudinal center of mass, friction and slot clearance vary. "
        "Success requires center within clearance + 3 mm in x/y, seated height "
        "within 8 mm and tilt at most 12 degrees."
    )
    requires_branching = True

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("mass_kg", 0.15, 0.45, nominal=0.25),
                Continuous("com_x", -0.035, 0.035, nominal=0.0),
                Continuous("friction", 0.30, 0.70, nominal=0.55),
                Continuous("slot_clearance", 0.004, 0.009, nominal=0.007),
                Continuous("distractor_roughness", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_color", ("blue", "green", "gray"), nominal="gray"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 14.0  # [6.0, 20.0]\n"
            "GRIP_OFFSET_X = 0.0  # [-0.035, 0.035]\n"
            "LIFT_SPEED = 0.06  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    b = api.get_object_pose("box")["pos"]\n'
            '    s = api.get_object_pose("slot")["pos"]\n'
            "    gx = b[0] + GRIP_OFFSET_X\n"
            "    tx = s[0] + GRIP_OFFSET_X\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([gx, b[1], b[2] + 0.07], speed=0.10)\n"
            "    api.move_to([gx, b[1], b[2]], speed=0.06)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([gx, b[1], b[2] + 0.08], speed=LIFT_SPEED)\n"
            "    api.move_to([tx, s[1], s[2] + 0.08], speed=0.08)\n"
            "    api.move_to([tx, s[1], s[2] + 0.024], speed=0.04)\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([tx, s[1], s[2] + 0.09], speed=0.08)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        mass_kg = float(theta["mass_kg"])
        com_x = float(theta["com_x"])
        friction = float(theta["friction"])
        clr = float(theta["slot_clearance"])
        hx, hy, hz = _BOX_HALF
        sx, sy = _SLOT_CENTER

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))

            # Slot marker body (for get_object_pose("slot")) + 4 guide walls
            slot = spec.worldbody.add_body(name="slot", pos=[sx, sy, TABLE_Z])
            wall_t = 0.008
            wall_hz = 0.5 * _SLOT_WALL_H
            wall_z = wall_hz
            # Y-side walls (left/right of slot)
            for side in (-1.0, 1.0):
                slot.add_geom(
                    name=f"slot_y_{'p' if side > 0 else 'n'}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.0, side * (hy + clr + wall_t), wall_z],
                    size=[hx + clr + 2.0 * wall_t, wall_t, wall_hz],
                    friction=[0.5, 0.005, 0.0001],
                    rgba=[0.35, 0.38, 0.44, 1.0],
                )
            # X-side walls (front/back of slot)
            for side in (-1.0, 1.0):
                slot.add_geom(
                    name=f"slot_x_{'p' if side > 0 else 'n'}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[side * (hx + clr + wall_t), 0.0, wall_z],
                    size=[wall_t, hy + clr, wall_hz],
                    friction=[0.5, 0.005, 0.0001],
                    rgba=[0.35, 0.38, 0.44, 1.0],
                )

            # Eccentric CoM box: inertiafromgeom=false requires explicit mass/ipos/inertia
            box = spec.worldbody.add_body(name="box", pos=list(_BOX_START))
            box.add_joint(name="box_free", type=mujoco.mjtJoint.mjJNT_FREE)
            box.mass = mass_kg
            box.ipos = [com_x, 0.0, 0.0]
            ixx = (1.0 / 12.0) * mass_kg * ((2 * hy) ** 2 + (2 * hz) ** 2)
            iyy = (1.0 / 12.0) * mass_kg * ((2 * hx) ** 2 + (2 * hz) ** 2)
            izz = (1.0 / 12.0) * mass_kg * ((2 * hx) ** 2 + (2 * hy) ** 2)
            box.inertia = [max(ixx, 1e-5), max(iyy, 1e-5), max(izz, 1e-5)]
            box.add_geom(
                name="box_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx, hy, hz],
                friction=[friction, 0.001, 0.0001],
                rgba=[0.78, 0.48, 0.22, 1.0],
            )

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            bid = ep.model.body("box").id
            pos = ep.data.xpos[bid]
            dx = abs(float(pos[0]) - sx)
            dy = abs(float(pos[1]) - sy)
            dz = abs(float(pos[2]) - (TABLE_Z + hz))
            tilt = body_tilt_rad(ep.data, bid)
            tilt_deg = math.degrees(tilt)

            inside_xy = dx <= clr + 0.003 and dy <= clr + 0.003
            flat_and_seated = dz <= 0.008 and tilt_deg <= 12.0
            if inside_xy and flat_and_seated:
                margin_xy = min(
                    (clr + 0.003 - dx) / max(clr + 0.003, 1e-4),
                    (clr + 0.003 - dy) / max(clr + 0.003, 1e-4),
                )
                margin_tilt = (12.0 - tilt_deg) / 12.0
                margin = max(0.0, min(margin_xy, margin_tilt))
                return EpisodeResult(success=True, score=0.25 + 0.75 * margin)

            err_norm = min(
                0.99,
                0.10 + 1.8 * math.hypot(dx, dy) + 0.35 * min(1.0, dz / 0.04) + 0.35 * min(1.0, tilt_deg / 90.0),
            )
            tag = "tilted" if tilt_deg > 12.0 else ("not_seated" if dz > 0.008 else "missed_slot")
            return EpisodeResult(success=False, score=-err_norm, failure=tag)

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        com_x = float(theta["com_x"])
        req_f = max(8.5, float(theta["mass_kg"]) * 9.81 / max(float(theta["friction"]), 0.2) * 0.75)
        f = min(17.5, req_f)
        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n"
            f"COM_SHIFT_X = {com_x:.5f}  # [-0.04, 0.04]\n"
            "LIFT_SPEED = 0.050  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    b = api.get_object_pose("box")["pos"]\n'
            '    s = api.get_object_pose("slot")["pos"]\n'
            "    gx = b[0] + COM_SHIFT_X\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([gx, b[1], b[2] + 0.07], speed=0.10)\n"
            "    api.move_to([gx, b[1], b[2]], speed=0.06)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([gx, b[1], b[2] + 0.08], speed=LIFT_SPEED)\n"
            "    tx = s[0] + COM_SHIFT_X\n"
            "    api.move_to([tx, s[1], s[2] + 0.08], speed=0.08)\n"
            "    api.move_to([tx, s[1], s[2] + 0.022], speed=0.03)\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([tx, s[1], s[2] + 0.09], speed=0.08)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.02  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    b = api.get_object_pose("box")["pos"]\n'
            '    s = api.get_object_pose("slot")["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([b[0], b[1], b[2] + 0.07], speed=0.05)\n"
            "    api.move_to([b[0], b[1], b[2]], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([b[0], b[1], b[2] + 0.08], speed=LIFT_SPEED)\n"
            "    api.move_to([s[0], s[1], s[2] + 0.08], speed=0.03)\n"
            "    api.move_to([s[0], s[1], s[2] + 0.025], speed=0.02)\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([s[0], s[1], s[2] + 0.09], speed=0.05)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        cx = float(theta["com_x"])
        if abs(cx) < 0.010:
            return "centered"
        return "front_heavy" if cx > 0.0 else "back_heavy"
