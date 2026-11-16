"""T6: Rotary valve with discrete lock state machine (free, press_first, lift_first)."""

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

_VALVE_XY = (0.35, 0.0)
_VALVE_Z = TABLE_Z + 0.050
_SPRING_K = 1200.0


class T6ValveTask(TaskBackend):
    """Unlock and rotate a valve handle >= 55 degrees."""

    name = "t6_valve"
    probe_config = ('valve', 'above', True)
    task_description = "Grasp the valve bar 'valve' and rotate counterclockwise at least 55 degrees. The detent may be free, require downward press or upward pull before rotation. Axial unlock resistance varies from 4-8 N."
    requires_branching = True

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Discrete("lock_mode", ("free", "press_first", "lift_first"), nominal="free"),
                Continuous("unlock_force_n", 4.0, 8.0, nominal=5.5),
                Continuous("valve_damping", 0.06, 0.20, nominal=0.10),
                Continuous("distractor_pipe_temp", 20.0, 80.0, nominal=40.0),
                Discrete("distractor_label", ("hot", "cold"), nominal="cold"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 14.0  # [1.0, 20.0]\n"
            "ROT_RAD = 1.30  # [0.8, 1.8]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    v = api.get_object_pose("valve")["pos"]\n'
            "    ee = api.get_ee_pose()\n"
            "    q = ee['quat']\n"
            "    c = math.cos(0.5 * ROT_RAD)\n"
            "    s = math.sin(0.5 * ROT_RAD)\n"
            "    q_rot = [c * q[0] - s * q[3], c * q[1] - s * q[2], c * q[2] + s * q[1], c * q[3] + s * q[0]]\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([v[0], v[1], v[2] + 0.06], speed=0.08)\n"
            "    api.move_to([v[0], v[1], v[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([v[0], v[1], v[2]], quat=q_rot, speed=0.06)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        lock_mode = str(theta["lock_mode"])
        unlock_f = float(theta["unlock_force_n"])
        damp = float(theta["valve_damping"])

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))
            spec.worldbody.add_geom(
                name="valve_pedestal",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[_VALVE_XY[0], _VALVE_XY[1], TABLE_Z + 0.015],
                size=[0.032, 0.015, 0.0],
                contype=0,
                conaffinity=0,
                rgba=[0.35, 0.38, 0.42, 1.0],
            )
            valve = spec.worldbody.add_body(name="valve", pos=[_VALVE_XY[0], _VALVE_XY[1], _VALVE_Z])
            valve.add_joint(
                name="valve_slide",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, 0, 1],
                range=[-0.009, 0.009],
                stiffness=_SPRING_K,
                damping=12.0,
            )
            valve.add_joint(
                name="valve_rot",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 0, 1],
                range=[-0.15, 2.2],
                damping=damp,
                frictionloss=0.03,
            )
            valve.mass = 0.18
            valve.ipos = [0.0, 0.0, 0.0]
            valve.inertia = [2e-4, 2e-4, 3e-4]
            valve.add_geom(
                name="valve_bar",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.036, 0.012, 0.014],
                friction=[1.5, 0.005, 0.0001],
                rgba=[0.82, 0.24, 0.20, 1.0],
            )

        state = {
            "init": False,
            "slide_q_adr": -1,
            "rot_q_adr": -1,
            "rot_v_adr": -1,
            "slide_q0": 0.0,
            "unlocked": (lock_mode == "free"),
            "dwell": 0,
        }

        def step_cb(ep: MujocoEpisode) -> str | None:
            if not state["init"]:
                sid = ep.model.joint("valve_slide").id
                rid = ep.model.joint("valve_rot").id
                state["slide_q_adr"] = int(ep.model.jnt_qposadr[sid])
                state["rot_q_adr"] = int(ep.model.jnt_qposadr[rid])
                state["rot_v_adr"] = int(ep.model.jnt_dofadr[rid])
                state["slide_q0"] = float(ep.data.qpos[state["slide_q_adr"]])
                state["init"] = True

            if not state["unlocked"]:
                ep.data.qpos[state["rot_q_adr"]] = 0.0
                ep.data.qvel[state["rot_v_adr"]] = 0.0
                dq = float(ep.data.qpos[state["slide_q_adr"]]) - state["slide_q0"]
                f_down = -_SPRING_K * dq
                f_up = _SPRING_K * dq
                cond = (lock_mode == "press_first" and f_down >= unlock_f) or (
                    lock_mode == "lift_first" and f_up >= unlock_f
                )
                if cond:
                    state["dwell"] += 1
                    if state["dwell"] >= 8:
                        state["unlocked"] = True
                else:
                    state["dwell"] = 0
            return None

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            rot_deg = math.degrees(float(ep.data.joint("valve_rot").qpos[0]))
            target_deg = 55.0
            if rot_deg >= target_deg:
                margin = min(1.0, (rot_deg - target_deg) / 30.0)
                return EpisodeResult(success=True, score=0.25 + 0.75 * margin)
            failure = "locked" if not state["unlocked"] else "under_rotated"
            return EpisodeResult(
                success=False,
                score=-min(0.95, 0.20 + (target_deg - max(0.0, rot_deg)) / target_deg * 0.75),
                failure=failure,
            )

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            step_callback=step_cb,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        mode = str(theta["lock_mode"])
        dz = -0.014 if mode == "press_first" else (0.014 if mode == "lift_first" else 0.0)
        unlock_block = ""
        if dz != 0.0:
            unlock_block = (
                f"    api.move_to([v[0], v[1], v[2] + ({dz:.5f})], speed=0.03)\n"
                "    api.move_to([v[0], v[1], v[2]], speed=0.04)\n"
            )
        return (
            "GRIP_FORCE = 16.0  # [1.0, 20.0]\n"
            "ROT_RAD = 1.35  # [0.8, 1.8]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    v = api.get_object_pose("valve")["pos"]\n'
            "    ee = api.get_ee_pose()\n"
            "    q = ee['quat']\n"
            "    c = math.cos(0.5 * ROT_RAD)\n"
            "    s = math.sin(0.5 * ROT_RAD)\n"
            "    q_rot = [c * q[0] - s * q[3], c * q[1] - s * q[2], c * q[2] + s * q[1], c * q[3] + s * q[0]]\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([v[0], v[1], v[2] + 0.06], speed=0.08)\n"
            "    api.move_to([v[0], v[1], v[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            f"{unlock_block}"
            "    api.move_to([v[0], v[1], v[2]], quat=q_rot, speed=0.06)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "ROT_RAD = 1.30  # [0.8, 1.8]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    v = api.get_object_pose("valve")["pos"]\n'
            "    ee = api.get_ee_pose()\n"
            "    q = ee['quat']\n"
            "    c = math.cos(0.5 * ROT_RAD)\n"
            "    s = math.sin(0.5 * ROT_RAD)\n"
            "    q_rot = [c * q[0] - s * q[3], c * q[1] - s * q[2], c * q[2] + s * q[1], c * q[3] + s * q[0]]\n"
            "    api.open_gripper(0.065)\n"
            "    api.move_to([v[0], v[1], v[2] + 0.06], speed=0.04)\n"
            "    api.move_to([v[0], v[1], v[2]], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([v[0], v[1], v[2]], quat=q_rot, speed=0.02)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return str(theta["lock_mode"])
