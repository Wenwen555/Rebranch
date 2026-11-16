"""T2: Fragile empty vs heavy sand-filled paper cup (real-world target)."""

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
    clamped_pad_force,
)

_CUP_POS = (0.35, 0.0, TABLE_Z + 0.035)
_CUP_RADIUS = 0.025
_CUP_HALF_H = 0.035
_HEAVY_YIELD_N = 17.0


class T2CupTask(TaskBackend):
    """Pick and lift a paper cup that may be empty/fragile or sand-filled/heavy."""

    name = "t2_cup"
    probe_config = ('cup', 'above', True)
    task_description = "Pick up the paper cup 'cup' and lift it at least 8 cm without crushing or dropping it. Cups vary between empty thin-walled and sand-filled, with different mass and contact compliance. Grip yield limits and friction vary; the usable gripper force range is 1-20 N."
    requires_branching = True

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Discrete("cup_type", ("empty", "sand"), nominal="empty"),
                Continuous("empty_yield_n", 4.0, 7.0, nominal=5.5),
                Continuous("sand_mass_kg", 0.30, 0.50, nominal=0.40),
                Continuous("friction", 0.22, 0.34, nominal=0.28),
                Continuous("distractor_rim_hue", 0.0, 1.0, nominal=0.4),
                Discrete("distractor_sleeve", ("plain", "ribbed"), nominal="plain"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 3.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.05  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("cup")["pos"]\n'
            "    api.open_gripper(0.068)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.11], speed=LIFT_SPEED)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        is_empty = str(theta["cup_type"]) == "empty"
        mass_kg = 0.015 if is_empty else float(theta["sand_mass_kg"])
        yield_n = float(theta["empty_yield_n"]) if is_empty else _HEAVY_YIELD_N
        friction = float(theta["friction"])
        solref = [0.070, 1.0] if is_empty else [0.005, 1.0]

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))
            cup = spec.worldbody.add_body(name="cup", pos=list(_CUP_POS))
            cup.add_joint(name="cup_free", type=mujoco.mjtJoint.mjJNT_FREE)
            cup.mass = mass_kg
            cup.ipos = [0.0, 0.0, -0.008 if not is_empty else 0.0]
            ixx = 0.25 * mass_kg * (_CUP_RADIUS**2) + (1.0 / 12.0) * mass_kg * ((2 * _CUP_HALF_H) ** 2)
            izz = 0.5 * mass_kg * (_CUP_RADIUS**2)
            cup.inertia = [max(ixx, 1e-5), max(ixx, 1e-5), max(izz, 1e-5)]
            cup.add_geom(
                name="cup_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[_CUP_RADIUS, _CUP_HALF_H, 0.0],
                friction=[friction, 0.005, 0.0001],
                priority=1,
                solref=solref,
                rgba=[0.92, 0.90, 0.85, 1.0] if is_empty else [0.76, 0.68, 0.55, 1.0],
            )

        cf_buf = np.zeros(6, dtype=float)
        state = {"cup_gid": -1, "over_steps": 0}

        def step_cb(ep: MujocoEpisode) -> str | None:
            if state["cup_gid"] < 0:
                state["cup_gid"] = ep.model.geom("cup_geom").id
            fn_l, fn_r = clamped_pad_force(ep, state["cup_gid"], cf_buf=cf_buf)
            # Require sustained bilateral squeeze above yield (ignores single-step impact spikes)
            if fn_l > yield_n and fn_r > yield_n:
                state["over_steps"] += 1
                if state["over_steps"] >= 5:
                    return "crushed"
            else:
                state["over_steps"] = 0
            return None

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            bid = ep.model.body("cup").id
            cup_z = float(ep.data.xpos[bid][2])
            lift_dz = cup_z - _CUP_POS[2]
            gs = ep.get_gripper_state()
            if lift_dz >= 0.080 and gs["grasped"]:
                margin = min(1.0, (lift_dz - 0.080) / 0.040)
                return EpisodeResult(success=True, score=0.30 + 0.70 * margin)
            if lift_dz < 0.015 and not gs["grasped"]:
                return EpisodeResult(success=False, score=-0.75, failure="slipped")
            return EpisodeResult(
                success=False,
                score=-max(0.15, min(0.9, 0.8 - 6.0 * max(0.0, lift_dz))),
                failure="not_lifted",
            )

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            step_callback=step_cb,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        if str(theta["cup_type"]) == "empty":
            f = min(float(theta["empty_yield_n"]) - 0.8, 3.0)
        else:
            req = float(theta["sand_mass_kg"]) * 9.81 / max(2.0 * float(theta["friction"]), 0.2)
            f = min(15.2, max(req + 3.5, 11.5))
        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.040  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("cup")["pos"]\n'
            "    api.open_gripper(0.068)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.11], speed=LIFT_SPEED)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.020  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("cup")["pos"]\n'
            "    api.open_gripper(0.068)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.05)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.11], speed=LIFT_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return str(theta["cup_type"])
