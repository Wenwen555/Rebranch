"""T5: Drawer pull with rail friction and lateral/axial jamming hook."""

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
    geom_pair_contact_force_world,
    gripper_pad_ids,
)

_DRAWER_X = 0.42
_DRAWER_Z = TABLE_Z + 0.075


class T5DrawerTask(TaskBackend):
    """Pull a skewed-rail drawer open without triggering lateral-force rail jamming."""

    name = "t5_drawer"
    probe_config = ('handle', 'front', True)
    task_description = "Grasp 'handle' and pull the drawer open at least 7.5 cm. Rail resistance varies from 3-10 N and its axis may be skewed by up to 14 degrees. Sustained excessive lateral-to-axial force can lock the rail."
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("rail_friction_n", 3.0, 10.0, nominal=4.0),
                Continuous("jam_ratio_thresh", 0.16, 0.50, nominal=0.45),
                Continuous("rail_skew_deg", -14.0, 14.0, nominal=0.0),
                Continuous("distractor_handle_shine", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_wood_grain", ("oak", "walnut", "birch"), nominal="oak"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 15.0  # [1.0, 20.0]\n"
            "PULL_DIST = 0.095  # [0.04, 0.14]\n"
            "PULL_SPEED = 0.045  # [0.01, 0.20]\n\n"
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
        rail_f = float(theta["rail_friction_n"])
        jam_thresh = float(theta["jam_ratio_thresh"])
        skew_rad = math.radians(float(theta["rail_skew_deg"]))
        u_ax = np.array([-math.cos(skew_rad), -math.sin(skew_rad), 0.0], dtype=float)

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))
            spec.worldbody.add_geom(
                name="drawer_housing",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[_DRAWER_X + 0.05, 0.0, _DRAWER_Z],
                size=[0.04, 0.095, 0.065],
                contype=0,
                conaffinity=0,
                rgba=[0.46, 0.40, 0.34, 1.0],
            )
            drawer = spec.worldbody.add_body(name="drawer", pos=[_DRAWER_X, 0.0, _DRAWER_Z])
            drawer.add_joint(
                name="drawer_slide",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=list(u_ax),
                range=[-0.005, 0.14],
                damping=12.0,
                frictionloss=rail_f,
            )
            drawer.mass = 0.50
            drawer.ipos = [0.02, 0.0, 0.0]
            drawer.inertia = [0.002, 0.002, 0.002]
            drawer.add_geom(
                name="drawer_front",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0, 0.0, 0.0],
                size=[0.008, 0.085, 0.055],
                contype=0,
                conaffinity=0,
                rgba=[0.66, 0.54, 0.42, 1.0],
            )
            drawer.add_geom(
                name="handle",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[-0.018, 0.0, 0.0],
                size=[0.012, 0.010, 0.022],
                friction=[1.5, 0.005, 0.0001],
                rgba=[0.20, 0.22, 0.26, 1.0],
            )

        cf_buf = np.zeros(6, dtype=float)
        state = {
            "init": False,
            "pad_l": -1,
            "pad_r": -1,
            "handle_g": -1,
            "q_adr": -1,
            "v_adr": -1,
            "jam_steps": 0,
            "locked": False,
            "lock_q": 0.0,
        }

        def step_cb(ep: MujocoEpisode) -> str | None:
            if not state["init"]:
                pad_l, pad_r = gripper_pad_ids(ep.model)
                state["pad_l"] = pad_l
                state["pad_r"] = pad_r
                state["handle_g"] = ep.model.geom("handle").id
                jid = ep.model.joint("drawer_slide").id
                state["q_adr"] = int(ep.model.jnt_qposadr[jid])
                state["v_adr"] = int(ep.model.jnt_dofadr[jid])
                state["init"] = True

            if state["locked"]:
                ep.data.qpos[state["q_adr"]] = state["lock_q"]
                ep.data.qvel[state["v_adr"]] = 0.0
                return None

            f_l = geom_pair_contact_force_world(
                ep.model, ep.data, state["handle_g"], state["pad_l"], cf_buf=cf_buf
            )
            f_r = geom_pair_contact_force_world(
                ep.model, ep.data, state["handle_g"], state["pad_r"], cf_buf=cf_buf
            )
            f_tot = f_l + f_r
            f_ax_signed = float(np.dot(f_tot, u_ax))
            if f_ax_signed > 3.0:
                f_xy = np.array([f_tot[0], f_tot[1], 0.0], dtype=float)
                f_lat = float(np.linalg.norm(f_xy - f_ax_signed * u_ax))
                if f_lat / f_ax_signed > jam_thresh:
                    state["jam_steps"] += 1
                    if state["jam_steps"] >= 8:
                        state["locked"] = True
                        state["lock_q"] = float(ep.data.qpos[state["q_adr"]])
                        ep.data.qvel[state["v_adr"]] = 0.0
                else:
                    state["jam_steps"] = 0
            else:
                state["jam_steps"] = 0
            return None

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            q = float(ep.data.joint("drawer_slide").qpos[0])
            target = 0.075
            if q >= target:
                margin = min(1.0, (q - target) / 0.035)
                return EpisodeResult(success=True, score=0.25 + 0.75 * margin)
            failure = "rail_jammed" if state["locked"] else "insufficient_pull"
            return EpisodeResult(
                success=False,
                score=-min(0.95, 0.15 + (target - q) / target * 0.8),
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
        skew_rad = math.radians(float(theta["rail_skew_deg"]))
        dx = -0.105 * math.cos(skew_rad)
        dy = -0.105 * math.sin(skew_rad)
        f = min(18.5, max(13.0, float(theta["rail_friction_n"]) + 7.5))
        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n"
            f"PULL_DX = {dx:.5f}  # [-0.14, -0.05]\n"
            f"PULL_DY = {dy:.5f}  # [-0.04, 0.04]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    h = api.get_object_pose("handle")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([h[0] - 0.06, h[1], h[2]], speed=0.08)\n"
            "    api.move_to([h[0], h[1], h[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            '    api.move_compliant([h[0] + PULL_DX, h[1] + PULL_DY, h[2]], free_axes="zXYZ", speed=0.04)\n'
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "PULL_SPEED = 0.020  # [0.01, 0.20]\n\n"
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
        deg = float(theta["rail_skew_deg"])
        if abs(deg) < 3.0:
            return "straight"
        return "skew_pos" if deg > 0.0 else "skew_neg"
