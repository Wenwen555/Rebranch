"""Real-world PiPER-X Episode sharing ProprioceptiveEstimator and control.py."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from rebranch.api.primitives import RobotAPI
from rebranch.api.proprioception import DEFAULT_PIPERX_XML, ProprioceptiveEstimator
from rebranch.real.piperx_driver import PiperXDriver
from rebranch.sim.backend import SNAPSHOT_KEYS, Episode, EpisodeResult
from rebranch.sim.control import (
    DEFAULT_ADM_K,
    Q_HOME,
    admittance_step,
    axis_angle_to_mat,
    dls_ik_step,
    orientation_error_world,
    parse_free_axes,
    quat_to_mat,
    quintic_blend,
)


class PiperXEpisode(Episode, RobotAPI):
    """Hardware RobotAPI episode backed by `PiperXDriver`, `ProprioceptiveEstimator`, and `control.py`."""

    def __init__(
        self,
        *,
        driver: PiperXDriver | None = None,
        model_path: str | Path | None = None,
        dt: float = 0.02,
        dry_run: bool = False,
        object_pose_fn: Callable[[str], dict[str, list[float]]] | None = None,
        evaluate_fn: Callable[[PiperXEpisode], EpisodeResult] | None = None,
    ) -> None:
        self.api: RobotAPI = self
        self.driver = driver if driver is not None else PiperXDriver(dry_run=dry_run)
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_PIPERX_XML
        self.dt = float(dt)
        self.sim_time = 0.0
        self._object_pose_fn = object_pose_fn
        self._evaluate_fn = evaluate_fn

        self._estimator = ProprioceptiveEstimator(model_path=self.model_path, tcp_site="tcp")
        self._ik_model = self._estimator.model
        self._ik_data = mujoco.MjData(self._ik_model)
        self._tcp_id = self._estimator.site_id
        self._jacp = np.zeros((3, self._ik_model.nv), dtype=float)
        self._jacr = np.zeros((3, self._ik_model.nv), dtype=float)
        self._k_adm = DEFAULT_ADM_K.copy()
        self._gripper_mode = "width"

        if not self.driver.connected:
            self.driver.connect()

        q, _, tau = self.driver.read_joints()
        if np.linalg.norm(q) < 1e-9:
            self.driver.send_joint_positions(Q_HOME)
        self.zero_ee_wrench()

    def _ik_sync(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ik_data.qpos[:6] = np.asarray(q, dtype=float).reshape(6)
        mujoco.mj_forward(self._ik_model, self._ik_data)
        pos = self._ik_data.site_xpos[self._tcp_id].copy()
        R = self._ik_data.site_xmat[self._tcp_id].reshape(3, 3).copy()
        return pos, R

    def get_ee_pose(self) -> dict[str, list[float]]:
        q, _, _ = self.driver.read_joints()
        pos, quat = self._estimator.fk(q)
        return {
            "pos": [float(v) for v in pos],
            "quat": [float(v) for v in quat],
        }

    def zero_ee_wrench(self) -> None:
        q, _, tau = self.driver.read_joints()
        self._estimator.tare(q, tau)

    def get_ee_wrench(self, frame: str = "world") -> dict[str, list[float]]:
        q, qd, tau = self.driver.read_joints()
        return self._estimator.estimate(q, qd, tau, frame=frame)

    def get_object_pose(self, name: str) -> dict[str, list[float]]:
        if self._object_pose_fn is not None:
            return self._object_pose_fn(name)
        raise RuntimeError(
            f"No real perception provider configured for get_object_pose({name!r})"
        )

    def open_gripper(self, width: float = 0.07) -> None:
        self._gripper_mode = "width"
        self.driver.send_gripper(float(np.clip(width, 0.0, 0.07)), effort_n=4.0)
        self.driver.sleep_dt(0.25)
        self.sim_time += 0.25

    def close_gripper(self, force: float = 8.0) -> None:
        self._gripper_mode = "force"
        self.driver.send_gripper(0.0, effort_n=float(np.clip(force, 1.0, 20.0)))
        self.driver.sleep_dt(0.30)
        self.sim_time += 0.30

    def get_gripper_state(self) -> dict[str, Any]:
        width, effort = self.driver.read_gripper()
        grasped = bool(self._gripper_mode == "force" and width > 0.002 and effort > 0.5)
        return {"width": float(width), "effort": float(effort), "grasped": grasped}

    def _execute_cartesian(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None,
        speed: float,
        *,
        free_axes: str = "",
        duration: float | None = None,
        max_force: float | None = None,
        max_torque: float | None = None,
    ) -> tuple[bool, dict[str, list[float]]]:
        q, _, _ = self.driver.read_joints()
        p0, R0 = self._ik_sync(q)
        p1 = np.asarray(pos, dtype=float).reshape(3)
        R1 = R0.copy() if quat is None else quat_to_mat(quat)

        dp = p1 - p0
        rot_delta = orientation_error_world(R1, R0)
        dist = float(np.linalg.norm(dp))
        rot_dist = float(np.linalg.norm(rot_delta))

        eff_speed = max(float(speed), 1e-4)
        eff_rot_speed = max(eff_speed * 4.0, 0.25)
        T = (
            max(float(duration), 0.05)
            if duration is not None
            else max(0.05, dist / eff_speed, rot_dist / eff_rot_speed)
        )
        n_steps = max(1, int(math.ceil(T / self.dt)))

        free_idx = parse_free_axes(free_axes)
        x_adm = p0.copy()
        R_adm = R0.copy()
        check_contact = max_force is not None and max_torque is not None
        contacted = False
        last_w = self.get_ee_wrench("world")

        for k in range(n_steps):
            u = (k + 1) / n_steps
            s, _, _ = quintic_blend(u)
            x_traj = p0 + s * dp
            R_traj = axis_angle_to_mat(s * rot_delta) @ R0

            q_cur, qd_cur, tau_cur = self.driver.read_joints()
            self._ik_sync(q_cur)
            if free_idx:
                last_w = self._estimator.estimate(q_cur, qd_cur, tau_cur, frame="world")
                w_vec = np.concatenate([last_w["force"], last_w["torque"]])
                x_next, R_next = admittance_step(
                    x_adm, R_adm, w_vec, free_idx, self._k_adm, self.dt
                )
                for idx in range(3):
                    if idx not in free_idx:
                        x_next[idx] = x_traj[idx]
                if not any(idx in free_idx for idx in (3, 4, 5)):
                    R_next = R_traj
                x_adm, R_adm = x_next, R_next
            else:
                x_adm, R_adm = x_traj, R_traj

            q_cmd = dls_ik_step(
                self._ik_model,
                self._ik_data,
                self._tcp_id,
                x_adm,
                R_adm,
                jacp=self._jacp,
                jacr=self._jacr,
            )
            self.driver.send_joint_positions(q_cmd)
            self.driver.sleep_dt(self.dt)
            self.sim_time += self.dt

            if check_contact:
                q_m, qd_m, tau_m = self.driver.read_joints()
                last_w = self._estimator.estimate(q_m, qd_m, tau_m, frame="world")
                f_norm = math.sqrt(sum(v * v for v in last_w["force"]))
                t_norm = math.sqrt(sum(v * v for v in last_w["torque"]))
                if f_norm >= float(max_force) or t_norm >= float(max_torque):
                    contacted = True
                    break

        return contacted, last_w

    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.08,
    ) -> None:
        self._execute_cartesian(pos, quat, speed)

    def move_until_contact(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.04,
        max_force: float = 6.0,
        max_torque: float = 0.8,
    ) -> dict[str, Any]:
        contacted, w = self._execute_cartesian(
            pos, quat, speed, max_force=max_force, max_torque=max_torque
        )
        ee = self.get_ee_pose()
        return {
            "contact": bool(contacted),
            "pos": ee["pos"],
            "force": w["force"],
            "torque": w["torque"],
        }

    def move_compliant(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        free_axes: str = "",
        speed: float = 0.04,
        duration: float | None = None,
    ) -> None:
        parse_free_axes(free_axes)
        self._execute_cartesian(pos, quat, speed, free_axes=free_axes, duration=duration)

    def snapshot(self) -> dict[str, float]:
        pose = self.get_ee_pose()
        w = self.get_ee_wrench("world")
        gs = self.get_gripper_state()
        snap = {
            "ee_x": float(pose["pos"][0]),
            "ee_y": float(pose["pos"][1]),
            "ee_z": float(pose["pos"][2]),
            "ee_qw": float(pose["quat"][0]),
            "ee_qx": float(pose["quat"][1]),
            "ee_qy": float(pose["quat"][2]),
            "ee_qz": float(pose["quat"][3]),
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
            "grip_width": float(gs["width"]),
            "grip_effort": float(gs["effort"]),
            "grip_grasped": 1.0 if gs["grasped"] else 0.0,
        }
        assert tuple(snap.keys()) == SNAPSHOT_KEYS
        return snap

    def evaluate(self) -> EpisodeResult:
        if self._evaluate_fn is not None:
            return self._evaluate_fn(self)
        return EpisodeResult(success=True, score=1.0)
