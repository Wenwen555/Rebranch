"""T8: Thin card pick with +/- 8 mm visual z_bias and table contact calibration."""

from __future__ import annotations

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
    gripper_pad_ids,
)

_CARD_HALF = (0.032, 0.022, 0.0055)  # 11 mm thick, 44 mm wide
_CARD_POS = (0.35, 0.0, TABLE_Z + _CARD_HALF[2])


class T8CardTask(TaskBackend):
    """Pick a thin card whose visual pose has +/- 8 mm z_bias."""

    name = "t8_card"
    probe_config = ('card', 'above', False)
    task_description = "Pick up the 11 mm thick flat card 'card' and lift it at least 7 cm. Its visual pose has unknown vertical bias within +/- 8 mm. Sustained gripper-table force above 14 N is a failure. Friction and mass vary."
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("z_bias", -0.008, 0.008, nominal=0.0),
                Continuous("card_friction", 0.55, 0.95, nominal=0.75),
                Continuous("mass_kg", 0.05, 0.12, nominal=0.06),
                Continuous("distractor_card_print", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_corner", ("round", "square"), nominal="round"),
            ]
        )
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 11.0  # [1.0, 20.0]\n"
            "Z_GRASP_OFFSET = 0.0080  # [0.004, 0.022]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("card")["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([c[0], c[1], c[2] + 0.06], speed=0.08)\n"
            "    api.move_to([c[0], c[1], c[2] + Z_GRASP_OFFSET], speed=0.04)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([c[0], c[1], c[2] + 0.11], speed=0.05)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        z_bias = float(theta["z_bias"])
        fric = float(theta["card_friction"])
        mass_kg = float(theta["mass_kg"])

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(0.75, 0.005, 0.0001))
            card = spec.worldbody.add_body(name="card", pos=list(_CARD_POS))
            card.add_joint(name="card_free", type=mujoco.mjtJoint.mjJNT_FREE)
            card.mass = mass_kg
            card.ipos = [0.0, 0.0, 0.0]
            card.inertia = [5e-5, 5e-5, 8e-5]
            card.add_geom(
                name="card_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(_CARD_HALF),
                friction=[fric, 0.005, 0.0001],
                rgba=[0.18, 0.56, 0.82, 1.0],
            )

        def obj_pose_fn(name: str, pose: dict[str, list[float]]) -> dict[str, list[float]]:
            if name == "card":
                out_pos = list(pose["pos"])
                out_pos[2] += z_bias
                return {"pos": out_pos, "quat": list(pose["quat"])}
            return pose

        cf_buf = np.zeros(6, dtype=float)
        state = {"init": False, "pad_l": -1, "pad_r": -1, "table_g": -1, "smash_steps": 0}

        def step_cb(ep: MujocoEpisode) -> str | None:
            if not state["init"]:
                state["pad_l"], state["pad_r"] = gripper_pad_ids(ep.model)
                state["table_g"] = ep.model.geom("table").id
                state["init"] = True
            fn = geom_pair_normal_force(
                ep.model, ep.data, state["pad_l"], state["table_g"], cf_buf=cf_buf
            ) + geom_pair_normal_force(
                ep.model, ep.data, state["pad_r"], state["table_g"], cf_buf=cf_buf
            )
            if fn > 14.0:
                state["smash_steps"] += 1
                if state["smash_steps"] >= 10:
                    return "table_collision"
            else:
                state["smash_steps"] = 0
            return None

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            if ep.safety_violation is not None:
                return EpisodeResult(
                    success=False,
                    score=-1.0,
                    failure=ep.safety_violation,
                    safety_violation=ep.safety_violation,
                )
            bid = ep.model.body("card").id
            card_z = float(ep.data.xpos[bid][2])
            lift_dz = card_z - _CARD_POS[2]
            gs = ep.get_gripper_state()
            if lift_dz >= 0.070 and gs["grasped"]:
                margin = min(1.0, (lift_dz - 0.070) / 0.040)
                return EpisodeResult(success=True, score=0.30 + 0.70 * margin)
            return EpisodeResult(
                success=False,
                score=-max(0.20, min(0.95, 0.85 - 8.0 * max(0.0, lift_dz))),
                failure="missed_card",
            )

        return MujocoEpisode(
            theta=theta,
            seed=seed,
            spec_mutator=mutate,
            step_callback=step_cb,
            object_pose_fn=obj_pose_fn,
            evaluate_fn=evaluate,
        )

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        _ = theta
        return (
            "GRIP_FORCE = 11.0  # [1.0, 20.0]\n"
            "Z_PROBE_OFFSET = -0.0016  # [-0.004, 0.006]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("card")["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([c[0], c[1] - 0.06, c[2] + 0.06], speed=0.08)\n"
            "    probe = api.move_until_contact([c[0], c[1] - 0.06, c[2] - 0.03], speed=0.04, max_force=4.0, max_torque=0.6)\n"
            "    z_grasp = probe['pos'][2] + Z_PROBE_OFFSET\n"
            "    api.move_to([c[0], c[1] - 0.06, z_grasp + 0.04], speed=0.08)\n"
            "    api.move_to([c[0], c[1], z_grasp + 0.04], speed=0.08)\n"
            "    api.move_to([c[0], c[1], z_grasp], speed=0.04)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([c[0], c[1], z_grasp + 0.10], speed=0.05)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "Z_GRASP_OFFSET = 0.0080  # [0.004, 0.022]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    c = api.get_object_pose("card")["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([c[0], c[1], c[2] + 0.06], speed=0.04)\n"
            "    api.move_to([c[0], c[1], c[2] + Z_GRASP_OFFSET], speed=0.02)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([c[0], c[1], c[2] + 0.11], speed=0.02)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        zb = float(theta["z_bias"])
        if abs(zb) < 0.0025:
            return "unbiased"
        return "high_bias" if zb > 0.0 else "low_bias"
