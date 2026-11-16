"""T7: Peg-in-hole insertion with collision-geom normal inflation and chamfer."""

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
    geom_pair_normal_force,
)

_PEG_START = (0.30, -0.09, TABLE_Z + 0.032)
_PEG_HALF = (0.012, 0.012, 0.032)
_HOLE_VIS = (0.38, 0.07, TABLE_Z)
_HOLE_INNER_HALF = 0.0154
_HOLE_DEPTH = 0.034


class T7PegTask(TaskBackend):
    """Pick a peg and insert it into a chamfered socket with hidden collision inflation."""

    name = "t7_peg"
    probe_config = ('peg', 'above', True)
    task_description = "Pick the 24 mm square peg 'peg' and insert it into socket 'hole' by at least 20 mm with lateral center error at most 5 mm. Collision half-width inflation is 0.2-1.5 mm; the true socket center differs from its visual pose by up to 8.5 mm per axis. Sustained socket contact force above 24 N is a failure."
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("inflation_mm", 0.2, 1.5, nominal=0.25),
                Continuous("hole_offset_x", -0.0085, 0.0085, nominal=0.0),
                Continuous("hole_offset_y", -0.0085, 0.0085, nominal=0.0),
                Continuous("peg_friction", 0.25, 0.60, nominal=0.35),
                Continuous("distractor_chamfer_color", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_material", ("aluminum", "delrin"), nominal="aluminum"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 14.0  # [1.0, 20.0]\n"
            "INSERT_SPEED = 0.04  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("peg")["pos"]\n'
            '    h = api.get_object_pose("hole")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.07], speed=0.10)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.09], speed=0.08)\n"
            "    api.move_to([h[0], h[1], h[2] + 0.095], speed=0.08)\n"
            "    api.move_to([h[0], h[1], h[2] + 0.032], speed=INSERT_SPEED)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        inf_m = float(theta["inflation_mm"]) * 1e-3
        ox = float(theta["hole_offset_x"])
        oy = float(theta["hole_offset_y"])
        fric = float(theta["peg_friction"])
        hx_true = _HOLE_VIS[0] + ox
        hy_true = _HOLE_VIS[1] + oy

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.70, 0.005, 0.0001))

            hole = spec.worldbody.add_body(name="hole", pos=[hx_true, hy_true, TABLE_Z])
            wt = 0.012
            wz = 0.5 * _HOLE_DEPTH
            r = _HOLE_INNER_HALF
            for gname, gpos, gsize in [
                ("hole_wall_xp", [r + wt, 0.0, wz], [wt, r + 2 * wt, wz]),
                ("hole_wall_xn", [-(r + wt), 0.0, wz], [wt, r + 2 * wt, wz]),
                ("hole_wall_yp", [0.0, r + wt, wz], [r, wt, wz]),
                ("hole_wall_yn", [0.0, -(r + wt), wz], [r, wt, wz]),
            ]:
                hole.add_geom(
                    name=gname,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=gpos,
                    size=gsize,
                    friction=[fric, 0.005, 0.0001],
                    rgba=[0.36, 0.40, 0.48, 1.0],
                )
            ch_z = _HOLE_DEPTH + 0.0018
            ch_off = r + 0.0018
            s45 = math.sin(math.radians(22.5))
            c45 = math.cos(math.radians(22.5))
            for name_s, pos_c, quat_c, size_c in [
                ("ch_xp", [ch_off, 0.0, ch_z], [c45, 0.0, -s45, 0.0], [0.0020, r + 0.006, 0.0020]),
                ("ch_xn", [-ch_off, 0.0, ch_z], [c45, 0.0, s45, 0.0], [0.0020, r + 0.006, 0.0020]),
                ("ch_yp", [0.0, ch_off, ch_z], [c45, s45, 0.0, 0.0], [r + 0.006, 0.0020, 0.0020]),
                ("ch_yn", [0.0, -ch_off, ch_z], [c45, -s45, 0.0, 0.0], [r + 0.006, 0.0020, 0.0020]),
            ]:
                hole.add_geom(
                    name=name_s,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos_c,
                    quat=quat_c,
                    size=size_c,
                    friction=[fric, 0.005, 0.0001],
                    rgba=[0.48, 0.52, 0.60, 1.0],
                )

            peg = spec.worldbody.add_body(name="peg", pos=list(_PEG_START))
            peg.add_joint(name="peg_free", type=mujoco.mjtJoint.mjJNT_FREE)
            peg.mass = 0.16
            peg.ipos = [0.0, 0.0, 0.0]
            peg.inertia = [1e-4, 1e-4, 1e-4]
            peg.add_geom(
                name="peg_visual",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(_PEG_HALF),
                contype=0,
                conaffinity=0,
                rgba=[0.84, 0.62, 0.22, 1.0],
            )
            peg.add_geom(
                name="peg_col",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[_PEG_HALF[0] + inf_m, _PEG_HALF[1] + inf_m, _PEG_HALF[2]],
                friction=[fric, 0.005, 0.0001],
                rgba=[0.84, 0.62, 0.22, 0.0],
            )

        def obj_pose_fn(name: str, pose: dict[str, list[float]]) -> dict[str, list[float]]:
            if name == "hole":
                return {
                    "pos": [float(_HOLE_VIS[0]), float(_HOLE_VIS[1]), float(_HOLE_VIS[2])],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                }
            return pose

        cf_buf = np.zeros(6, dtype=float)
        hole_gnames = (
            "hole_wall_xp",
            "hole_wall_xn",
            "hole_wall_yp",
            "hole_wall_yn",
            "ch_xp",
            "ch_xn",
            "ch_yp",
            "ch_yn",
        )
        state = {"init": False, "peg_g": -1, "hole_gs": [], "over_steps": 0}

        def step_cb(ep: MujocoEpisode) -> str | None:
            if not state["init"]:
                state["peg_g"] = ep.model.geom("peg_col").id
                state["hole_gs"] = [ep.model.geom(n).id for n in hole_gnames]
                state["init"] = True
            f_hole = sum(
                geom_pair_normal_force(ep.model, ep.data, state["peg_g"], hg, cf_buf=cf_buf)
                for hg in state["hole_gs"]
            )
            if f_hole > 24.0:
                state["over_steps"] += 1
                if state["over_steps"] >= 10:
                    return "jammed_excessive_force"
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
            bid = ep.model.body("peg").id
            pos = ep.data.xpos[bid]
            dxy = math.hypot(float(pos[0]) - hx_true, float(pos[1]) - hy_true)
            peg_bottom_z = float(pos[2]) - _PEG_HALF[2]
            inserted_depth = TABLE_Z + _HOLE_DEPTH - peg_bottom_z

            if dxy <= 0.005 and inserted_depth >= 0.020:
                margin = min(1.0, (inserted_depth - 0.020) / 0.014)
                return EpisodeResult(success=True, score=0.25 + 0.75 * margin)
            err = min(0.95, 0.15 + 5.0 * dxy + max(0.0, 0.020 - inserted_depth) * 15.0)
            return EpisodeResult(success=False, score=-err, failure="jammed_at_rim")

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            step_callback=step_cb,
            object_pose_fn=obj_pose_fn,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        ox = float(theta["hole_offset_x"])
        oy = float(theta["hole_offset_y"])
        return (
            "GRIP_FORCE = 14.0  # [1.0, 20.0]\n"
            f"OFF_X = {ox:.5f}  # [-0.010, 0.010]\n"
            f"OFF_Y = {oy:.5f}  # [-0.010, 0.010]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("peg")["pos"]\n'
            '    h = api.get_object_pose("hole")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.07], speed=0.10)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.09], speed=0.08)\n"
            "    api.move_to([h[0] + OFF_X, h[1] + OFF_Y, h[2] + 0.095], speed=0.08)\n"
            "    api.move_to([h[0] + OFF_X, h[1] + OFF_Y, h[2] + 0.032], speed=0.03)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "INSERT_SPEED = 0.02  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    p = api.get_object_pose("peg")["pos"]\n'
            '    h = api.get_object_pose("hole")["pos"]\n'
            "    api.open_gripper(0.06)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.07], speed=0.05)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.09], speed=0.04)\n"
            "    api.move_to([h[0], h[1], h[2] + 0.095], speed=0.04)\n"
            "    api.move_to([h[0], h[1], h[2] + 0.032], speed=INSERT_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return "tight_fit" if float(theta["inflation_mm"]) > 0.85 else "loose_fit"
