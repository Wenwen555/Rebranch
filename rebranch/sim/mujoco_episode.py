"""MuJoCo simulation Episode implementing the frozen 10-primitive RobotAPI."""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import mujoco
import numpy as np

from rebranch.api.noise import JointNoiseModel
from rebranch.api.primitives import EpisodeTerminated, RobotAPI
from rebranch.api.proprioception import DEFAULT_PIPERX_XML, ProprioceptiveEstimator
from rebranch.sim.backend import SNAPSHOT_KEYS, Episode, EpisodeResult
from rebranch.sim.render import render_mujoco_rgb_png
from rebranch.sim.physcap_mass_reveal import PhysCaPMassReveal
from rebranch.sim.control import (
    CONTROL_MODES,
    DEFAULT_ADM_K,
    DEFAULT_OSC_KD,
    DEFAULT_OSC_KP,
    Q_HOME,
    admittance_step,
    axis_angle_to_mat,
    gripper_control_torques,
    mat_to_quat,
    orientation_error_world,
    osc_torque_6dof,
    parse_free_axes,
    pos_admittance_torques,
    quat_to_mat,
    quintic_blend,
)

# v2 changes action timing; old runs retain their v1 interpretation.
SIM_PROTOCOL_VERSION = "rebranch-mujoco-v2"


class MujocoEpisode(Episode, RobotAPI):
    """Parameterizable MuJoCo rollout episode for PiPER-X manipulation tasks."""

    def __init__(
        self,
        *,
        theta: dict[str, Any] | None = None,
        seed: int = 0,
        model_path: str | Path | None = None,
        spec_mutator: Callable[[mujoco.MjSpec], None] | None = None,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
        q_home: Sequence[float] = Q_HOME,
        init_gripper_width: float = 0.07,
        control_mode: Literal["mit_osc", "pos_admittance"] = "mit_osc",
        noise_enabled: bool = True,
        calib_path: str | Path | None = None,
        step_callback: Callable[..., str | None] | None = None,
        evaluate_fn: Callable[[MujocoEpisode], EpisodeResult] | None = None,
        object_pose_fn: Callable[[str, dict[str, list[float]]], dict[str, list[float]]] | None = None,
        tcp_site: str = "tcp",
        settle_steps: int = 50,
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
    ) -> None:
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unsupported control_mode {control_mode!r}; expected {CONTROL_MODES}")

        self.api: RobotAPI = self
        self._theta: dict[str, Any] = dict(theta) if theta is not None else {}
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_PIPERX_XML
        self.control_mode = control_mode
        self.tcp_site = tcp_site
        self._evaluate_fn = evaluate_fn
        self._object_pose_fn = object_pose_fn
        self._physcap_mass_reveal: PhysCaPMassReveal | None = None
        self.mass_measurements: list[dict[str, Any]] = []

        if model is not None:
            self.model = model
            self.data = data if data is not None else mujoco.MjData(model)
        else:
            spec = mujoco.MjSpec.from_file(str(self.model_path))
            if spec_mutator is not None:
                spec_mutator(spec)
            self.model = spec.compile()
            self.data = mujoco.MjData(self.model)

        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if self.tcp_id < 0:
            raise ValueError(f"Site {tcp_site!r} not found in compiled MuJoCo model")
        self.dt = float(self.model.opt.timestep)

        self._kp = DEFAULT_OSC_KP.copy() if kp is None else np.asarray(kp, dtype=float).reshape(6)
        self._kd = DEFAULT_OSC_KD.copy() if kd is None else np.asarray(kd, dtype=float).reshape(6)
        self._k_adm = DEFAULT_ADM_K.copy()

        # Preallocated scratch arrays for 500 Hz control
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)
        self._M_full = np.zeros((self.model.nv, self.model.nv), dtype=float)

        # Proprioceptive estimator (always uses unloaded nominal robot XML, never true contact force)
        self._estimator = ProprioceptiveEstimator(model_path=self.model_path, tcp_site=tcp_site)
        self._noise = JointNoiseModel(calib_path=calib_path, enabled=noise_enabled)
        self._noise.reset(self._rng)

        self._step_callback = step_callback
        self._cb_arity = 0
        if step_callback is not None:
            try:
                sig = inspect.signature(step_callback)
                pos_params = [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
                self._cb_arity = 3 if has_varargs else len(pos_params)
            except (TypeError, ValueError):
                self._cb_arity = 2

        self._safety_violation: str | None = None
        self._terminated_reason: str | None = None

        # Reset arm and gripper to home pose
        q_init = np.asarray(q_home, dtype=float).reshape(6)
        w_init = float(np.clip(init_gripper_width, 0.0, 0.07))
        self.data.qpos[:6] = q_init
        self.data.qpos[6] = 0.5 * w_init
        self.data.qpos[7] = -0.5 * w_init
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._x_des = self.data.site_xpos[self.tcp_id].copy()
        self._R_des = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        self._xd_dot = np.zeros(3, dtype=float)
        self._wd_des = np.zeros(3, dtype=float)
        self._xdd_des = np.zeros(3, dtype=float)
        self._wdd_des = np.zeros(3, dtype=float)

        self._gripper_mode: Literal["width", "force"] = "width"
        self._target_width = w_init
        self._target_force = 0.0

        self._free_axes: str = ""
        self._compliance_frame: Literal["world", "ee"] = "world"
        self._x_adm = self._x_des.copy()
        self._R_adm = self._R_des.copy()

        self._last_tau_meas = np.zeros(6, dtype=float)
        self._last_grip_width = w_init
        self._last_grip_effort = 0.0

        # Initial settle without user step_callback so resting contacts stabilize
        saved_cb = self._step_callback
        self._step_callback = None
        try:
            for _ in range(max(1, int(settle_steps))):
                self._step_once(update_wrench=False)
        finally:
            self._step_callback = saved_cb

        self.sim_time = 0.0
        self.zero_ee_wrench()

    protocol_version = SIM_PROTOCOL_VERSION

    @property
    def theta(self) -> dict[str, Any]:
        """Internal ground-truth parameter dict (for task evaluation hooks only)."""
        return self._theta

    @property
    def safety_violation(self) -> str | None:
        return self._safety_violation

    @property
    def terminated_reason(self) -> str | None:
        return self._terminated_reason

    def _invoke_step_callback(self) -> None:
        if self._step_callback is None:
            return
        try:
            if self._cb_arity == 0:
                res = self._step_callback()
            elif self._cb_arity == 1:
                res = self._step_callback(self)
            elif self._cb_arity == 2:
                res = self._step_callback(self.model, self.data)
            else:
                res = self._step_callback(self.model, self.data, self)
        except EpisodeTerminated as exc:
            self._terminated_reason = exc.reason
            raise
        if isinstance(res, str) and res:
            self._safety_violation = res
            self._terminated_reason = res
            raise EpisodeTerminated(res)

    def _step_once(self, *, update_wrench: bool = False) -> dict[str, list[float]] | None:
        """Advance physics and control by one 500 Hz timestep."""
        if self.control_mode == "mit_osc":
            v_des = np.concatenate([self._xd_dot, self._wd_des])
            a_des = np.concatenate([self._xdd_des, self._wdd_des])
            tau_arm = osc_torque_6dof(
                self.model,
                self.data,
                self.tcp_id,
                self._x_des,
                self._R_des,
                kp=self._kp,
                kd=self._kd,
                xd_dot=v_des,
                xdd_des=a_des,
                free_axes=self._free_axes,
                compliance_frame=self._compliance_frame,
                jacp=self._jacp,
                jacr=self._jacr,
                M_full=self._M_full,
            )
        else:
            if self._free_axes:
                w_cur = self._estimator.estimate(
                    self.data.qpos[:6], self.data.qvel[:6], self._last_tau_meas, frame="world"
                )
                w_vec = np.concatenate([w_cur["force"], w_cur["torque"]])
                free_idx = parse_free_axes(self._free_axes)
                x_next, R_next = admittance_step(
                    self._x_adm, self._R_adm, w_vec, self._free_axes, self._k_adm, self.dt
                )
                for idx in range(3):
                    if idx not in free_idx:
                        x_next[idx] = self._x_des[idx]
                if not any(idx in free_idx for idx in (3, 4, 5)):
                    R_next = self._R_des.copy()
                self._x_adm = x_next
                self._R_adm = R_next
            else:
                self._x_adm = self._x_des.copy()
                self._R_adm = self._R_des.copy()
            tau_arm = pos_admittance_torques(
                self.model,
                self.data,
                self.tcp_id,
                self._x_adm,
                self._R_adm,
                jacp=self._jacp,
                jacr=self._jacr,
            )

        tau_grip = gripper_control_torques(
            self.model,
            self.data,
            self._gripper_mode,
            target_width=self._target_width,
            target_force=self._target_force,
        )
        self.data.ctrl[:6] = tau_arm
        self.data.ctrl[6:8] = tau_grip

        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        if self._physcap_mass_reveal is not None:
            self._physcap_mass_reveal.update()

        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self._safety_violation = "numerical_instability"
            self._terminated_reason = "numerical_instability"
            raise EpisodeTerminated("numerical_instability")

        # Sample noisy SDK measurements (only from actuator/joint states, never true contact forces)
        self._last_tau_meas = self._noise.apply_arm(
            self.data.qpos[:6],
            self.data.qvel[:6],
            self.data.qfrc_actuator[:6],
            self._rng,
        )
        w_true = float(self.data.qpos[6] - self.data.qpos[7])
        eff_true = 0.5 * (
            abs(float(self.data.qfrc_actuator[6])) + abs(float(self.data.qfrc_actuator[7]))
        )
        self._last_grip_width, self._last_grip_effort = self._noise.apply_gripper(
            w_true, eff_true, self._rng
        )

        self._invoke_step_callback()

        if update_wrench:
            return self._estimator.estimate(
                self.data.qpos[:6], self.data.qvel[:6], self._last_tau_meas, frame="world"
            )
        return None

    def step_idle(self, steps: int) -> None:
        """Hold current setpoint for `steps` simulation steps."""
        self._xd_dot[:] = 0.0
        self._wd_des[:] = 0.0
        self._xdd_des[:] = 0.0
        self._wdd_des[:] = 0.0
        for _ in range(int(steps)):
            self._step_once(update_wrench=False)

    # ------------------------------------------------------------------
    # RobotAPI (10 frozen primitives)
    # ------------------------------------------------------------------

    def get_ee_pose(self) -> dict[str, list[float]]:
        pos = self.data.site_xpos[self.tcp_id]
        quat = mat_to_quat(self.data.site_xmat[self.tcp_id].reshape(3, 3))
        return {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        }

    def zero_ee_wrench(self) -> None:
        self._estimator.tare(self.data.qpos[:6], self._last_tau_meas)

    def get_ee_wrench(self, frame: str = "world") -> dict[str, list[float]]:
        return self._estimator.estimate(
            self.data.qpos[:6], self.data.qvel[:6], self._last_tau_meas, frame=frame
        )

    def get_object_pose(self, name: str) -> dict[str, list[float]]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            p = self.data.xpos[bid]
            q = self.data.xquat[bid]
        else:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                p = self.data.geom_xpos[gid]
                q = mat_to_quat(self.data.geom_xmat[gid].reshape(3, 3))
            else:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid < 0:
                    raise KeyError(f"Scene object {name!r} not found")
                p = self.data.site_xpos[sid]
                q = mat_to_quat(self.data.site_xmat[sid].reshape(3, 3))
        pose = {
            "pos": [float(p[0]), float(p[1]), float(p[2])],
            "quat": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        }
        if self._object_pose_fn is not None:
            pose = self._object_pose_fn(name, pose)
        return pose

    def open_gripper(self, width: float = 0.07) -> None:
        self._gripper_mode = "width"
        self._target_width = float(np.clip(width, 0.0, 0.07))
        self._target_force = 0.0
        self.step_idle(int(math.ceil(0.8 / self.dt)))

    def close_gripper(self, force: float = 8.0) -> None:
        self._gripper_mode = "force"
        self._target_force = float(np.clip(force, 1.0, 20.0))
        # Include contact relaxation, not just the first impact/compression.
        self.step_idle(int(math.ceil(0.8 / self.dt)))

    def get_gripper_state(self) -> dict[str, Any]:
        grasped = bool(
            self._gripper_mode == "force"
            and self._last_grip_width > 0.002
            and self._last_grip_effort > 0.5
        )
        return {
            "width": float(self._last_grip_width),
            "effort": float(self._last_grip_effort),
            "grasped": grasped,
        }

    def _run_trajectory(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None,
        speed: float,
        *,
        free_axes: str = "",
        duration: float | None = None,
        max_force: float | None = None,
        max_torque: float | None = None,
        settle_steps: int = 40,
    ) -> tuple[bool, dict[str, list[float]]]:
        p0 = self._x_des.copy()
        R0 = self._R_des.copy()
        p1 = np.asarray(pos, dtype=float).reshape(3)
        R1 = R0.copy() if quat is None else quat_to_mat(quat)

        dp = p1 - p0
        rot_delta = orientation_error_world(R1, R0)
        dist = float(np.linalg.norm(dp))
        rot_dist = float(np.linalg.norm(rot_delta))

        eff_speed = max(float(speed), 1e-4)
        eff_rot_speed = max(eff_speed * 4.0, 0.25)
        if duration is not None:
            T = max(float(duration), 0.05)
        else:
            # A 1 mm move must not become a 50 ms acceleration impulse.
            T = max(0.25, dist / eff_speed, rot_dist / eff_rot_speed)
        traj_steps = max(1, int(math.ceil(T / self.dt)))

        self._free_axes = free_axes
        self._x_adm = p0.copy()
        self._R_adm = R0.copy()
        check_contact = max_force is not None and max_torque is not None
        contacted = False
        last_w: dict[str, list[float]] | None = None

        try:
            total_steps = traj_steps + max(settle_steps, int(math.ceil(0.2 / self.dt)))
            for k in range(total_steps):
                if k < traj_steps:
                    u = (k + 1) / traj_steps
                    s, ds_du, dds_du2 = quintic_blend(u)
                    ds = ds_du / T
                    dds = dds_du2 / (T * T)

                    self._x_des = p0 + s * dp
                    self._xd_dot = ds * dp
                    self._xdd_des = dds * dp
                    self._R_des = axis_angle_to_mat(s * rot_delta) @ R0
                    self._wd_des = ds * rot_delta
                    self._wdd_des = dds * rot_delta
                else:
                    self._x_des = p1.copy()
                    self._R_des = R1.copy()
                    self._xd_dot[:] = 0.0
                    self._wd_des[:] = 0.0
                    self._xdd_des[:] = 0.0
                    self._wdd_des[:] = 0.0

                w = self._step_once(update_wrench=check_contact)
                if w is not None:
                    last_w = w
                    f_norm = math.sqrt(sum(v * v for v in w["force"]))
                    t_norm = math.sqrt(sum(v * v for v in w["torque"]))
                    if f_norm >= float(max_force) or t_norm >= float(max_torque):
                        self._x_des = self.data.site_xpos[self.tcp_id].copy()
                        self._R_des = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
                        contacted = True
                        break
        finally:
            had_free = bool(self._free_axes)
            self._free_axes = ""
            self._xd_dot[:] = 0.0
            self._wd_des[:] = 0.0
            self._xdd_des[:] = 0.0
            self._wdd_des[:] = 0.0
            if had_free:
                self._x_des = self.data.site_xpos[self.tcp_id].copy()
                self._R_des = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

        if contacted:
            self.step_idle(20)
            last_w = self.get_ee_wrench("world")
        elif last_w is None:
            last_w = self.get_ee_wrench("world")
        return contacted, last_w

    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.08,
    ) -> None:
        self._run_trajectory(pos, quat, speed, settle_steps=40)

    def move_until_contact(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.04,
        max_force: float = 6.0,
        max_torque: float = 0.8,
    ) -> dict[str, Any]:
        contacted, w = self._run_trajectory(
            pos,
            quat,
            speed,
            max_force=max_force,
            max_torque=max_torque,
            settle_steps=0,
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
        # An explicit no-op duration is a hold, not a new trajectory to the
        # measured (load-deflected) pose. Preserve the controller's setpoint.
        if (duration is not None and not free_axes
                and np.linalg.norm(np.asarray(pos) - self.data.site_xpos[self.tcp_id]) < 1e-6
                and (quat is None or np.linalg.norm(orientation_error_world(
                    quat_to_mat(quat), self.data.site_xmat[self.tcp_id].reshape(3, 3))) < 1e-6)):
            self.step_idle(max(1, int(math.ceil(float(duration) / self.dt))))
            return
        self._run_trajectory(
            pos,
            quat,
            speed,
            free_axes=free_axes,
            duration=duration,
            settle_steps=25,
        )

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

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

    def render_rgb(self, *, width: int = 640, height: int = 480) -> bytes:
        """Render the current scene without advancing MuJoCo time."""
        return render_mujoco_rgb_png(
            self.model, self.data, width=width, height=height
        )

    def enable_physcap_mass_reveal(self) -> None:
        """Enable the appendix's simulator-only mass gate for this baseline."""
        self._physcap_mass_reveal = PhysCaPMassReveal(self.model, self.data)

    def get_mass(self, name: str) -> float | None:
        """Read gated simulator mass; None means grasp/lift evidence is incomplete."""
        if self._physcap_mass_reveal is None:
            raise RuntimeError("PhysCaP mass-reveal adapter is not enabled for this episode")
        value = self._physcap_mass_reveal.read(name)
        self.mass_measurements.append({
            "object_name": str(name),
            "value_kg": value,
            "gate": self._physcap_mass_reveal.state(str(name)),
            "sim_time_s": float(self.sim_time),
        })
        return value

    def evaluate(self) -> EpisodeResult:
        if self._evaluate_fn is not None:
            return self._evaluate_fn(self)
        if self._safety_violation is not None:
            return EpisodeResult(
                success=False,
                score=-1.0,
                failure=self._safety_violation,
                safety_violation=self._safety_violation,
            )
        if self._terminated_reason is not None:
            return EpisodeResult(
                success=False,
                score=-1.0,
                failure=self._terminated_reason,
            )
        return EpisodeResult(success=True, score=1.0)
