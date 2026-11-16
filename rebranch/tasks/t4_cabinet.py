"""T4: Desktop mini-cabinet with 4 articulation topologies (real-world target)."""

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
)

_DOOR_X = 0.42
_DOOR_W = 0.18
_DOOR_H = 0.15
_DOOR_Z_CENTER = TABLE_Z + 0.010 + 0.5 * _DOOR_H  # 0.105 m
_HANDLE_OFFSET_X = -0.018


class T4CabinetTask(TaskBackend):
    """Open a desktop mini-cabinet whose articulation topology is unknown."""

    name = "t4_cabinet"
    probe_config = ('handle', 'front', True)
    task_description = "Grasp 'handle' and open the cabinet: at least 30 degrees for a hinge or 6.5 cm for a sliding drawer. Articulation may be a left hinge, right hinge, bottom flap or drawer; resistance and damping vary."
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Discrete(
                    "topology",
                    ("left_hinge", "right_hinge", "bottom_flap", "drawer"),
                    nominal="drawer",
                ),
                Continuous("pull_resistance_n", 3.0, 8.0, nominal=4.5),
                Continuous("damping", 0.3, 1.0, nominal=0.5),
                Continuous("distractor_paint_gloss", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_knob_shape", ("bar", "round"), nominal="bar"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 14.0  # [1.0, 20.0]\n"
            "PULL_DIST = 0.09  # [0.03, 0.14]\n"
            "PULL_SPEED = 0.04  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    h = api.get_object_pose("handle")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([h[0] - 0.06, h[1], h[2]], speed=0.08)\n"
            "    api.move_to([h[0], h[1], h[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([h[0] - PULL_DIST, h[1], h[2]], speed=PULL_SPEED)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        topo = str(theta["topology"])
        res_n = float(theta["pull_resistance_n"])
        damp = float(theta["damping"])
        half_w = 0.5 * _DOOR_W
        half_h = 0.5 * _DOOR_H

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))

            # Non-colliding cabinet frame visual behind the door
            spec.worldbody.add_geom(
                name="cabinet_frame",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[_DOOR_X + 0.05, 0.0, _DOOR_Z_CENTER],
                size=[0.04, half_w + 0.01, half_h + 0.01],
                contype=0,
                conaffinity=0,
                rgba=[0.50, 0.44, 0.38, 1.0],
            )

            if topo == "left_hinge":
                door = spec.worldbody.add_body(name="door", pos=[_DOOR_X, half_w, _DOOR_Z_CENTER])
                door.add_joint(
                    name="cab_joint",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 0, 1],
                    range=[-1.35, 0.02],
                    damping=damp,
                    frictionloss=res_n * half_w,
                )
                local_center = [0.0, -half_w, 0.0]
            elif topo == "right_hinge":
                door = spec.worldbody.add_body(name="door", pos=[_DOOR_X, -half_w, _DOOR_Z_CENTER])
                door.add_joint(
                    name="cab_joint",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 0, 1],
                    range=[-0.02, 1.35],
                    damping=damp,
                    frictionloss=res_n * half_w,
                )
                local_center = [0.0, half_w, 0.0]
            elif topo == "bottom_flap":
                door = spec.worldbody.add_body(
                    name="door", pos=[_DOOR_X, 0.0, _DOOR_Z_CENTER - half_h]
                )
                door.add_joint(
                    name="cab_joint",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 1, 0],
                    range=[-1.35, 0.02],
                    damping=damp,
                    frictionloss=res_n * half_h,
                )
                local_center = [0.0, 0.0, half_h]
            else:
                door = spec.worldbody.add_body(name="door", pos=[_DOOR_X, 0.0, _DOOR_Z_CENTER])
                door.add_joint(
                    name="cab_joint",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[-1, 0, 0],
                    range=[-0.005, 0.14],
                    damping=damp * 25.0,
                    frictionloss=res_n,
                )
                local_center = [0.0, 0.0, 0.0]

            door.mass = 0.45
            door.ipos = local_center
            door.inertia = [0.0015, 0.0015, 0.0015]
            door.add_geom(
                name="door_panel",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=local_center,
                size=[0.008, half_w - 0.005, half_h - 0.005],
                contype=0,
                conaffinity=0,
                rgba=[0.66, 0.56, 0.44, 1.0],
            )
            handle_pos = [
                local_center[0] + _HANDLE_OFFSET_X,
                local_center[1],
                local_center[2],
            ]
            door.add_geom(
                name="handle",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=handle_pos,
                size=[0.012, 0.010, 0.022],
                friction=[1.5, 0.005, 0.0001],
                rgba=[0.22, 0.24, 0.28, 1.0],
            )

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            q = float(ep.data.joint("cab_joint").qpos[0])
            if topo == "drawer":
                opened = q
                target = 0.065
            else:
                opened = abs(q)
                target = math.radians(30.0)

            if opened >= target:
                margin = min(1.0, (opened - target) / max(0.5 * target, 1e-4))
                return EpisodeResult(success=True, score=0.25 + 0.75 * margin)
            deficit = (target - opened) / target
            return EpisodeResult(
                success=False,
                score=-min(0.95, 0.15 + 0.80 * deficit),
                failure="jammed_or_not_opened",
            )

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        topo = str(theta["topology"])
        f = min(18.5, max(12.0, float(theta["pull_resistance_n"]) + 7.5))
        if topo == "drawer":
            pull_stmt = '    api.move_to([h[0] - 0.105, h[1], h[2]], speed=0.04)\n'
        elif topo in ("left_hinge", "right_hinge"):
            pull_stmt = '    api.move_compliant([h[0] - 0.125, h[1], h[2]], free_axes="yzXYZ", speed=0.04)\n'
        else:
            pull_stmt = '    api.move_compliant([h[0] - 0.115, h[1], h[2]], free_axes="yzXYZ", speed=0.04)\n'

        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    h = api.get_object_pose("handle")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([h[0] - 0.06, h[1], h[2]], speed=0.08)\n"
            "    api.move_to([h[0], h[1], h[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            f"{pull_stmt}"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "PULL_SPEED = 0.02  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    h = api.get_object_pose("handle")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([h[0] - 0.06, h[1], h[2]], speed=0.04)\n"
            "    api.move_to([h[0], h[1], h[2]], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([h[0] - 0.10, h[1], h[2]], speed=PULL_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return str(theta["topology"])
