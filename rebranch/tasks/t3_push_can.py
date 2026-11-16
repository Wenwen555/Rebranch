"""T3: Tall can lateral push without tipping (natural MuJoCo tipping physics)."""

from __future__ import annotations

import math
from typing import Any

import mujoco

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

_CAN_RADIUS = 0.028
_CAN_HALF_H = 0.040  # 8 cm tall can
_CAN_START = (0.34, -0.07, TABLE_Z + _CAN_HALF_H)


class T3PushCanTask(TaskBackend):
    """Push a tall can laterally across the table without tipping it over."""

    name = "t3_push_can"
    probe_config = ('can', 'above', False)
    task_description = (
        "Push the tall cylinder 'can' (8 cm tall, diameter 56 mm) laterally along +y "
        "by at least 7.5 cm without tipping it over (tilt < 18 deg). "
        "Can mass (150-400 g), CoM height, and base friction vary across instances; "
        "pushing too high tips the can over when base friction is high."
    )
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("mass_kg", 0.15, 0.40, nominal=0.22),
                Continuous("com_z_offset", -0.025, 0.030, nominal=-0.015),
                Continuous("base_friction", 0.22, 0.70, nominal=0.25),
                Continuous("distractor_label_hue", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_brand", ("cola", "tonic", "soda"), nominal="soda"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "PUSH_HEIGHT_OFFSET = 0.068  # [0.028, 0.080]\n"
            "PUSH_SPEED = 0.045  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("can")["pos"]\n'
            "    z_push = 0.020 + PUSH_HEIGHT_OFFSET\n"
            "    api.open_gripper(0.00)\n"
            "    api.move_to([c[0], c[1] - 0.070, 0.140], speed=0.10)\n"
            "    api.move_to([c[0], c[1] - 0.070, z_push], speed=0.06)\n"
            "    api.move_to([c[0], c[1] + 0.11, z_push], speed=PUSH_SPEED)\n"
            "    api.move_to([c[0], c[1] + 0.11, 0.140], speed=0.08)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        mass_kg = float(theta["mass_kg"])
        com_z = float(theta["com_z_offset"])
        mu = float(theta["base_friction"])

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(mu, 0.005, 0.0001))
            can = spec.worldbody.add_body(name="can", pos=list(_CAN_START))
            can.add_joint(name="can_free", type=mujoco.mjtJoint.mjJNT_FREE)
            can.mass = mass_kg
            can.ipos = [0.0, 0.0, com_z]
            ixx = 0.25 * mass_kg * (_CAN_RADIUS**2) + (1.0 / 12.0) * mass_kg * ((2 * _CAN_HALF_H) ** 2)
            izz = 0.5 * mass_kg * (_CAN_RADIUS**2)
            can.inertia = [max(ixx, 1e-5), max(ixx, 1e-5), max(izz, 1e-5)]
            can.add_geom(
                name="can_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[_CAN_RADIUS, _CAN_HALF_H, 0.0],
                friction=[mu, 0.002, 0.0001],
                rgba=[0.85, 0.22, 0.20, 1.0],
            )

        state = {"can_bid": -1, "max_tilt_deg": 0.0}

        def step_cb(ep: MujocoEpisode) -> str | None:
            if state["can_bid"] < 0:
                state["can_bid"] = ep.model.body("can").id
            tilt_deg = math.degrees(body_tilt_rad(ep.data, state["can_bid"]))
            if tilt_deg > state["max_tilt_deg"]:
                state["max_tilt_deg"] = tilt_deg
            return None

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            bid = ep.model.body("can").id
            cy = float(ep.data.xpos[bid][1])
            final_tilt = math.degrees(body_tilt_rad(ep.data, bid))
            peak_tilt = max(state["max_tilt_deg"], final_tilt)
            disp_y = cy - _CAN_START[1]

            if peak_tilt >= 18.0:
                return EpisodeResult(
                    success=False,
                    score=-min(1.0, 0.2 + peak_tilt / 90.0),
                    failure="tipped",
                )
            if disp_y >= 0.075:
                tilt_margin = max(0.0, (18.0 - peak_tilt) / 18.0)
                return EpisodeResult(success=True, score=0.20 + 0.80 * tilt_margin)
            return EpisodeResult(
                success=False,
                score=-min(0.9, 0.2 + max(0.0, 0.075 - disp_y) * 5.0),
                failure="short_push",
            )

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            step_callback=step_cb,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        _ = theta
        return (
            "PUSH_HEIGHT_OFFSET = 0.028  # [0.028, 0.080]\n"
            "PUSH_SPEED = 0.010  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("can")["pos"]\n'
            "    z_push = 0.020 + PUSH_HEIGHT_OFFSET\n"
            "    api.open_gripper(0.00)\n"
            "    api.move_to([c[0], c[1] - 0.070, 0.140], speed=0.10)\n"
            "    api.move_to([c[0], c[1] - 0.070, z_push], speed=0.06)\n"
            "    api.move_to([c[0], c[1] + 0.11, z_push], speed=PUSH_SPEED)\n"
            "    api.move_to([c[0], c[1] + 0.11, 0.140], speed=0.08)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "PUSH_HEIGHT_OFFSET = 0.068  # [0.028, 0.080]\n"
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "PUSH_SPEED = 0.020  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("can")["pos"]\n'
            "    z_push = 0.020 + PUSH_HEIGHT_OFFSET\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([c[0], c[1] - 0.070, 0.140], speed=0.04)\n"
            "    api.move_to([c[0], c[1] - 0.070, z_push], speed=0.02)\n"
            "    api.move_to([c[0], c[1] + 0.11, z_push], speed=PUSH_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return "high_friction" if float(theta["base_friction"]) > 0.40 else "low_friction"
