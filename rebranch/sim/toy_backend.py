"""Analytical MuJoCo-free toy tasks for M0 algorithm verification."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from rebranch.api.primitives import EpisodeTerminated, RobotAPI
from rebranch.sim.backend import SNAPSHOT_KEYS, Episode, EpisodeResult, TaskBackend
from rebranch.sim.params import Continuous, Discrete, ParamSpace


_DEFAULT_OBS_SCALE: dict[str, float] = {
    "ee_x": 0.005,
    "ee_y": 0.005,
    "ee_z": 0.005,
    "ee_qw": 0.02,
    "ee_qx": 0.02,
    "ee_qy": 0.02,
    "ee_qz": 0.02,
    "fx": 0.3,
    "fy": 0.3,
    "fz": 0.3,
    "tx": 0.05,
    "ty": 0.05,
    "tz": 0.05,
    "jd1": 0.08,
    "jd2": 0.08,
    "jd3": 0.08,
    "jd4": 0.05,
    "jd5": 0.05,
    "jd6": 0.05,
    "grip_width": 0.001,
    "grip_effort": 0.2,
    "grip_grasped": 0.1,
}

_DEFAULT_OBS_API_MAP: dict[str, str] = {
    "ee_x": "api.get_ee_pose()['pos'][0]",
    "ee_y": "api.get_ee_pose()['pos'][1]",
    "ee_z": "api.get_ee_pose()['pos'][2]",
    "fx": "api.get_ee_wrench()['force'][0]",
    "fy": "api.get_ee_wrench()['force'][1]",
    "fz": "api.get_ee_wrench()['force'][2]",
    "tx": "api.get_ee_wrench()['torque'][0]",
    "ty": "api.get_ee_wrench()['torque'][1]",
    "tz": "api.get_ee_wrench()['torque'][2]",
    "jd1": "api.get_ee_wrench()['joint_delta'][0]",
    "jd2": "api.get_ee_wrench()['joint_delta'][1]",
    "jd3": "api.get_ee_wrench()['joint_delta'][2]",
    "grip_width": "api.get_gripper_state()['width']",
    "grip_effort": "api.get_gripper_state()['effort']",
    "grip_grasped": "api.get_gripper_state()['grasped']",
}


class _ToyPickAPI(RobotAPI):
    """Lightweight analytical RobotAPI for pick-and-lift toy tasks."""

    def __init__(
        self,
        *,
        obj_name: str,
        obj_pos: list[float],
        mass_kg: float,
        friction: float,
        yield_force_n: float,
        fragile_wall: bool,
        rng: np.random.Generator,
    ) -> None:
        self.obj_name = obj_name
        self.table_z = obj_pos[2]
        self.obj_pos = list(obj_pos)
        self.obj_quat = [1.0, 0.0, 0.0, 0.0]
        self.mass_kg = float(mass_kg)
        self.friction = float(friction)
        self.yield_force_n = float(yield_force_n)
        self.fragile_wall = bool(fragile_wall)
        self.rng = rng

        self.ee_pos = [0.30, 0.0, 0.18]
        self.ee_quat = [0.0, 1.0, 0.0, 0.0]
        self.grip_width = 0.07
        self.grip_effort = 0.0
        self.grip_force = 0.0
        self.grasped = False
        self.crushed = False
        self.slipped = False
        self.last_lift_speed = 0.08
        self.sim_time = 0.0
        self.wrench_tare = np.zeros(6, dtype=float)
        self._raw_bias = rng.normal(0.0, 0.05, size=6)
        self._last_wrench = self._compute_raw_wrench()

    def _compute_raw_wrench(self) -> np.ndarray:
        w = self._raw_bias.copy()
        if self.grasped and self.obj_pos[2] > self.table_z + 1e-4:
            w[2] += -self.mass_kg * 9.81
        return w

    def _current_wrench(self) -> np.ndarray:
        noise = self.rng.normal(0.0, [0.04, 0.04, 0.04, 0.005, 0.005, 0.005])
        return self._compute_raw_wrench() - self.wrench_tare + noise

    def get_ee_pose(self) -> dict[str, list[float]]:
        return {"pos": list(self.ee_pos), "quat": list(self.ee_quat)}

    def zero_ee_wrench(self) -> None:
        self.wrench_tare = self._compute_raw_wrench().copy()
        self.sim_time += 0.05

    def get_ee_wrench(self, frame: str = "world") -> dict[str, list[float]]:
        if frame not in ("world", "ee"):
            raise ValueError(f"Unknown frame: {frame!r}")
        w = self._current_wrench()
        self._last_wrench = w
        # Approximate joint delta proportional to wrench magnitude
        jd = [float(w[2] * 0.12), float(w[2] * 0.18), float(-w[2] * 0.15), 0.0, 0.0, 0.0]
        if frame == "ee":
            # Top-down EE has z pointing down: flip y,z signs wrt world
            return {
                "force": [float(w[0]), float(-w[1]), float(-w[2])],
                "torque": [float(w[3]), float(-w[4]), float(-w[5])],
                "joint_delta": jd,
            }
        return {
            "force": [float(v) for v in w[:3]],
            "torque": [float(v) for v in w[3:]],
            "joint_delta": jd,
        }

    def get_object_pose(self, name: str) -> dict[str, list[float]]:
        if name != self.obj_name:
            raise KeyError(f"Unknown object {name!r}")
        return {"pos": list(self.obj_pos), "quat": list(self.obj_quat)}

    def open_gripper(self, width: float = 0.07) -> None:
        w = min(max(float(width), 0.0), 0.07)
        self.grip_width = w
        self.grip_effort = 0.0
        self.grip_force = 0.0
        self.grasped = False
        self.obj_pos[2] = self.table_z
        self.sim_time += 0.2

    def close_gripper(self, force: float = 8.0) -> None:
        f = min(max(float(force), 1.0), 20.0)
        self.sim_time += 0.25
        dist = math.hypot(
            self.ee_pos[0] - self.obj_pos[0],
            self.ee_pos[1] - self.obj_pos[1],
        )
        dz = abs(self.ee_pos[2] - self.obj_pos[2])
        if dist < 0.03 and dz < 0.03:
            if f > self.yield_force_n:
                self.crushed = True
                self.grasped = False
                self.grip_width = 0.02
                self.grip_effort = f
                raise EpisodeTerminated("crushed")
            self.grasped = True
            self.slipped = False
            self.grip_force = f
            self.grip_effort = f
            # Fragile thin-walled objects visibly compress more at low force.
            compliance = 0.0025 if self.fragile_wall else 0.0002
            self.grip_width = max(0.02, 0.055 - compliance * f)
        else:
            self.grasped = False
            self.grip_force = 0.0
            self.grip_effort = 0.0
            self.grip_width = 0.0

    def get_gripper_state(self) -> dict[str, Any]:
        return {
            "width": float(self.grip_width),
            "effort": float(self.grip_effort),
            "grasped": bool(self.grasped),
        }

    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.08,
    ) -> None:
        target = [float(pos[0]), float(pos[1]), float(pos[2])]
        spd = max(float(speed), 1e-3)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, self.ee_pos)))
        self.sim_time += dist / spd

        if self.grasped and target[2] > self.obj_pos[2] + 1e-4:
            self.last_lift_speed = spd
            req = self.mass_kg * 9.81 * (1.0 + 1.5 * spd) / max(self.friction, 0.05)
            if self.grip_force >= req:
                self.obj_pos = list(target)
            else:
                self.grasped = False
                self.slipped = True
                self.obj_pos[2] = self.table_z

        self.ee_pos = target
        if quat is not None:
            self.ee_quat = [float(v) for v in quat]

    def move_until_contact(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.04,
        max_force: float = 6.0,
        max_torque: float = 0.8,
    ) -> dict[str, Any]:
        self.move_to(pos, quat=quat, speed=speed)
        w = self._current_wrench()
        return {
            "contact": False,
            "pos": list(self.ee_pos),
            "force": [float(v) for v in w[:3]],
            "torque": [float(v) for v in w[3:]],
        }

    def move_compliant(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        free_axes: str = "",
        speed: float = 0.04,
        duration: float | None = None,
    ) -> None:
        for ch in free_axes:
            if ch not in "xyzXYZ":
                raise ValueError(f"Invalid free_axes char {ch!r}")
        self.move_to(pos, quat=quat, speed=speed)
        if duration is not None:
            self.sim_time += float(duration)


class _ToyPickEpisode(Episode):
    """Episode wrapper enforcing SNAPSHOT_KEYS and bounded EpisodeResult."""

    def __init__(self, api: _ToyPickAPI) -> None:
        self.api = api

    @property
    def sim_time(self) -> float:
        return self.api.sim_time

    def snapshot(self) -> dict[str, float]:
        w = self.api.get_ee_wrench("world")
        snap = {
            "ee_x": float(self.api.ee_pos[0]),
            "ee_y": float(self.api.ee_pos[1]),
            "ee_z": float(self.api.ee_pos[2]),
            "ee_qw": float(self.api.ee_quat[0]),
            "ee_qx": float(self.api.ee_quat[1]),
            "ee_qy": float(self.api.ee_quat[2]),
            "ee_qz": float(self.api.ee_quat[3]),
            "fx": float(w["force"][0]),
            "fy": float(w["force"][1]),
            "fz": float(w["force"][2]),
            "tx": float(w["torque"][0]),
            "ty": float(w["torque"][1]),
            "tz": float(w["torque"][2]),
            "jd1": float(w["joint_delta"][0]),
            "jd2": float(w["joint_delta"][1]),
            "jd3": float(w["joint_delta"][2]),
            "jd4": float(w["joint_delta"][3]),
            "jd5": float(w["joint_delta"][4]),
            "jd6": float(w["joint_delta"][5]),
            "grip_width": float(self.api.grip_width),
            "grip_effort": float(self.api.grip_effort),
            "grip_grasped": 1.0 if self.api.grasped else 0.0,
        }
        assert tuple(snap.keys()) == SNAPSHOT_KEYS
        return snap

    def evaluate(self) -> EpisodeResult:
        if self.api.crushed:
            return EpisodeResult(
                success=False,
                score=-1.0,
                failure="crushed",
                safety_violation="crushed",
            )
        req = (
            self.api.mass_kg
            * 9.81
            * (1.0 + 1.5 * self.api.last_lift_speed)
            / max(self.api.friction, 0.05)
        )
        if self.api.grasped and self.api.obj_pos[2] >= self.api.table_z + 0.08:
            slip_margin = (self.api.grip_force - req) / 10.0
            yield_margin = (self.api.yield_force_n - self.api.grip_force) / 10.0
            margin = min(slip_margin, yield_margin)
            score = max(0.05, min(1.0, 0.25 + 0.75 * margin))
            return EpisodeResult(success=True, score=score)
        if self.api.slipped:
            deficit = max(0.0, req - self.api.grip_force)
            score = -min(0.95, 0.15 + deficit / 10.0)
            return EpisodeResult(success=False, score=score, failure="slipped")
        return EpisodeResult(success=False, score=-0.5, failure="not_lifted")


class ToyRetuneTask(TaskBackend):
    probe_config = ("block", "above", True)
    """Analytical task solvable by constant retuning without branching."""

    name = "toy_retune"
    task_description = (
        "Pick and lift the object 'block' by at least 10 cm without slipping or crushing it."
    )
    requires_branching = False

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Continuous("mass_kg", 0.05, 0.35, nominal=0.10),
                Continuous("friction", 0.45, 0.90, nominal=0.75),
                Continuous("yield_force_n", 12.0, 19.5, nominal=16.0),
                Continuous("distractor_hue", 0.0, 1.0, nominal=0.5),
                Discrete("distractor_label", ("a", "b", "c"), nominal="a"),
            ]
        )
        self.obs_scale = dict(_DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(_DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 3.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.12  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("block")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.10)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        rng = np.random.default_rng(seed)
        api = _ToyPickAPI(
            obj_name="block",
            obj_pos=[0.35, 0.0, 0.04],
            mass_kg=float(theta["mass_kg"]),
            friction=float(theta["friction"]),
            yield_force_n=float(theta["yield_force_n"]),
            fragile_wall=False,
            rng=rng,
        )
        return _ToyPickEpisode(api)

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        req = float(theta["mass_kg"]) * 9.81 * 1.03 / max(float(theta["friction"]), 0.05)
        f = min(float(theta["yield_force_n"]) - 0.5, max(req + 1.5, 9.5))
        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.020  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("block")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.10)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.05)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.02  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("block")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.05)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.03)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return "uniform_block"


class ToyBranchTask(TaskBackend):
    probe_config = ("cup", "above", True)
    """Analytical task requiring sensory probing and branching (fragile vs heavy)."""

    name = "toy_branch"
    task_description = (
        "Pick and lift the object 'cup' by at least 10 cm. Cups may be fragile thin-walled "
        "(low yield threshold, compliant width, light) or sturdy heavy (high slip threshold, rigid)."
    )
    requires_branching = True

    def __init__(self) -> None:
        self.param_space = ParamSpace(
            [
                Discrete("mode", ("fragile", "heavy"), nominal="fragile"),
                Continuous("yield_force_n", 4.0, 6.5, nominal=5.5),
                Continuous("heavy_mass_kg", 0.32, 0.50, nominal=0.40),
                Continuous("friction", 0.35, 0.55, nominal=0.45),
                Continuous("distractor_temp", 15.0, 35.0, nominal=22.0),
                Discrete("distractor_finish", ("matte", "glossy"), nominal="matte"),
            ]
        )
        self.obs_scale = dict(_DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(_DEFAULT_OBS_API_MAP)
        self.draft_policy = (
            "GRIP_FORCE = 3.5  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.04  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("cup")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        rng = np.random.default_rng(seed)
        is_fragile = theta["mode"] == "fragile"
        mass_kg = 0.04 if is_fragile else float(theta["heavy_mass_kg"])
        yield_force = float(theta["yield_force_n"]) if is_fragile else 19.5
        api = _ToyPickAPI(
            obj_name="cup",
            obj_pos=[0.35, 0.0, 0.04],
            mass_kg=mass_kg,
            friction=float(theta["friction"]),
            yield_force_n=yield_force,
            fragile_wall=is_fragile,
            rng=rng,
        )
        return _ToyPickEpisode(api)

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        if theta["mode"] == "fragile":
            f = min(float(theta["yield_force_n"]) - 0.5, 3.2)
        else:
            req = float(theta["heavy_mass_kg"]) * 9.81 * 1.03 / max(float(theta["friction"]), 0.05)
            f = min(18.5, max(req + 1.5, 16.0))
        return (
            f"GRIP_FORCE = {f:.3f}  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.020  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("cup")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.08)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.04)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def conservative_policy(self) -> str:
        return (
            "GRIP_FORCE = 20.0  # [1.0, 20.0]\n"
            "LIFT_SPEED = 0.02  # [0.01, 0.20]\n\n"
            "def run(api):\n"
            "    api.zero_ee_wrench()\n"
            '    pose = api.get_object_pose("cup")\n'
            '    p = pose["pos"]\n'
            "    api.open_gripper(0.065)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.08], speed=0.05)\n"
            "    api.move_to([p[0], p[1], p[2]], speed=0.03)\n"
            "    api.close_gripper(force=GRIP_FORCE)\n"
            "    api.move_to([p[0], p[1], p[2] + 0.12], speed=LIFT_SPEED)\n"
        )

    def latent_mode(self, theta: dict[str, Any]) -> str:
        return str(theta["mode"])
