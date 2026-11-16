"""6-DOF Operational Space Control (OSC), position admittance, and gripper control for PiPER-X."""
from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

import mujoco
import numpy as np

_MUJOCO_MAJOR_VERSION = int(mujoco.__version__.split(".", 1)[0])

Q_HOME = np.array([0.0, 1.48, -1.38, 1.30, 0.0, 0.0], dtype=float)
CONTROL_MODES = ("mit_osc", "pos_admittance")

# Default acceleration-level OSC gains (Lambda-scaled, critically damped at 500 Hz)
DEFAULT_OSC_KP = np.array([900.0, 900.0, 900.0, 400.0, 400.0, 400.0], dtype=float)
DEFAULT_OSC_KD = np.array([60.0, 60.0, 60.0, 40.0, 40.0, 40.0], dtype=float)

# Default admittance virtual stiffness (linear N/m, angular N*m/rad)
DEFAULT_ADM_K = np.array([250.0, 250.0, 250.0, 25.0, 25.0, 25.0], dtype=float)

_CHAR_TO_AXIS = {
    "x": 0,
    "y": 1,
    "z": 2,
    "X": 3,
    "Y": 4,
    "Z": 5,
    "rx": 3,
    "ry": 4,
    "rz": 5,
}


def mj_full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData, dst: np.ndarray) -> None:
    """Compute dense mass matrix compatible with both MuJoCo 4.x and 3.x."""
    if _MUJOCO_MAJOR_VERSION >= 4:
        mujoco.mj_fullM(model, data, dst)
    else:
        mujoco.mj_fullM(model, dst, data.qM)


def quat_to_mat(quat_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion must be non-zero")
    q = np.ascontiguousarray(q / norm, dtype=float)
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot, dtype=float).ravel())
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def orientation_error_world(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation error vector (axis * angle, rad) from R_cur to R_des."""
    R_err = R_des @ R_cur.T
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_err, dtype=float).ravel())
    if quat[0] < 0.0:
        quat *= -1.0
    sin_half = float(np.linalg.norm(quat[1:]))
    if sin_half < 1e-10:
        return 2.0 * quat[1:]
    angle = 2.0 * math.atan2(sin_half, float(quat[0]))
    return (angle / sin_half) * quat[1:]


def axis_angle_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Convert a 3D rotation vector (axis * angle, rad) to a 3x3 rotation matrix."""
    rv = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3, dtype=float)
    quat = np.zeros(4, dtype=float)
    mat = np.zeros(9, dtype=float)
    mujoco.mju_axisAngle2Quat(quat, rv / angle, angle)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def parse_free_axes(free_axes: str | Iterable[str] | set[int]) -> set[int]:
    """Parse compliant axes specification ('xyzXYZ' or axis index set) into {0..5}."""
    if isinstance(free_axes, set):
        for idx in free_axes:
            if not isinstance(idx, (int, np.integer)) or not 0 <= int(idx) <= 5:
                raise ValueError(f"Invalid axis index: {idx}")
        return {int(i) for i in free_axes}
    if isinstance(free_axes, str):
        tokens = list(free_axes)
    else:
        tokens = list(free_axes)
    indices: set[int] = set()
    for tok in tokens:
        if tok not in _CHAR_TO_AXIS:
            raise ValueError(
                f"Invalid free_axes token {tok!r}; expected characters in 'xyzXYZ'"
            )
        indices.add(_CHAR_TO_AXIS[tok])
    return indices


def quintic_blend(u: float) -> tuple[float, float, float]:
    """Return minimum-jerk normalized (s, ds/du, d^2s/du^2) for u in [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    s = u2 * u * (10.0 - 15.0 * u + 6.0 * u2)
    ds_du = 30.0 * u2 * (1.0 - 2.0 * u + u2)
    dds_du2 = 60.0 * u * (1.0 - 3.0 * u + 2.0 * u2)
    return s, ds_du, dds_du2


def osc_torque_6dof(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_des: np.ndarray,
    R_des: np.ndarray,
    kp: Sequence[float] | np.ndarray = DEFAULT_OSC_KP,
    kd: Sequence[float] | np.ndarray = DEFAULT_OSC_KD,
    *,
    xd_dot: np.ndarray | None = None,
    xdd_des: np.ndarray | None = None,
    free_axes: str | Iterable[str] | set[int] = "",
    compliance_frame: Literal["world", "ee"] = "world",
    kd_free_pos: float = 2.0,
    kd_free_rot: float = 0.05,
    damping_reg: float = 1e-4,
    n_arm: int = 6,
    jacp: np.ndarray | None = None,
    jacr: np.ndarray | None = None,
    M_full: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 6-DOF Operational Space Control joint torques (no 7-DOF nullspace).

    Uses dynamically consistent operational-space inertia with damped inverse
    regularization, zeros stiffness on `free_axes` while leaving light viscous
    damping, and adds `data.qfrc_bias[:n_arm]` for gravity/Coriolis compensation.
    """
    if compliance_frame not in ("world", "ee"):
        raise ValueError(f"Unsupported compliance_frame: {compliance_frame!r}")

    kp_arr = np.asarray(kp, dtype=float)
    kd_arr = np.asarray(kd, dtype=float)
    if kp_arr.shape != (6,) or kd_arr.shape != (6,):
        raise ValueError("kp and kd must be 6-element vectors")

    x_cur = data.site_xpos[site_id]
    R_cur = data.site_xmat[site_id].reshape(3, 3)

    if jacp is None:
        jacp = np.zeros((3, model.nv), dtype=float)
    if jacr is None:
        jacr = np.zeros((3, model.nv), dtype=float)
    if M_full is None:
        M_full = np.zeros((model.nv, model.nv), dtype=float)

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :n_arm], jacr[:, :n_arm]])

    mj_full_mass_matrix(model, data, M_full)
    M_arm = M_full[:n_arm, :n_arm] + 1e-6 * np.eye(n_arm)
    M_arm_inv = np.linalg.inv(M_arm)
    Lambda = np.linalg.inv(J @ M_arm_inv @ J.T + damping_reg * np.eye(6))

    err = np.concatenate([np.asarray(x_des, dtype=float) - x_cur, orientation_error_world(R_des, R_cur)])
    v_cur = J @ data.qvel[:n_arm]
    v_des = np.zeros(6, dtype=float) if xd_dot is None else np.asarray(xd_dot, dtype=float).copy()
    a_des = np.zeros(6, dtype=float) if xdd_des is None else np.asarray(xdd_des, dtype=float).copy()

    free_indices = parse_free_axes(free_axes)
    if not free_indices:
        F_task = Lambda @ (a_des + kp_arr * err + kd_arr * (v_des - v_cur))
    else:
        kp_loc = kp_arr.copy()
        kd_loc = kd_arr.copy()
        for idx in free_indices:
            kp_loc[idx] = 0.0
            kd_loc[idx] = 0.0

        if compliance_frame == "ee":
            R6 = np.zeros((6, 6), dtype=float)
            R6[:3, :3] = R_cur
            R6[3:, 3:] = R_cur
            err_loc = R6.T @ err
            v_cur_loc = R6.T @ v_cur
            v_des_loc = R6.T @ v_des
            a_des_loc = R6.T @ a_des
            for idx in free_indices:
                v_des_loc[idx] = 0.0
                a_des_loc[idx] = 0.0
            acc_cmd = R6 @ (a_des_loc + kp_loc * err_loc + kd_loc * (v_des_loc - v_cur_loc))
            F_loc = R6.T @ (Lambda @ acc_cmd)
            for idx in free_indices:
                F_loc[idx] = -(kd_free_pos if idx < 3 else kd_free_rot) * v_cur_loc[idx]
            F_task = R6 @ F_loc
        else:
            for idx in free_indices:
                v_des[idx] = 0.0
                a_des[idx] = 0.0
            F_task = Lambda @ (a_des + kp_loc * err + kd_loc * (v_des - v_cur))
            for idx in free_indices:
                F_task[idx] = -(kd_free_pos if idx < 3 else kd_free_rot) * v_cur[idx]

    tau = J.T @ F_task + data.qfrc_bias[:n_arm]
    ctrl_min = model.actuator_ctrlrange[:n_arm, 0]
    ctrl_max = model.actuator_ctrlrange[:n_arm, 1]
    return np.clip(tau, ctrl_min, ctrl_max)


def admittance_step(
    x_ref: np.ndarray,
    R_ref: np.ndarray,
    w_hat: np.ndarray,
    free_axes: str | Iterable[str] | set[int],
    k_adm: float | Sequence[float] | np.ndarray = DEFAULT_ADM_K,
    dt: float = 0.002,
    *,
    max_lin_vel: float = 0.25,
    max_ang_vel: float = 1.5,
    deadband: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance compliant reference pose (x_ref, R_ref) using estimated wrench w_hat.

    Along `free_axes`, velocity yields proportionally to `w_hat / b_adm`, where
    `b_adm = max(0.1 * k_adm, 0.5)`. Non-free axes follow `(x_ref, R_ref)` unchanged.
    """
    x_out = np.asarray(x_ref, dtype=float).copy()
    R_out = np.asarray(R_ref, dtype=float).copy()
    w = np.asarray(w_hat, dtype=float)
    if w.shape != (6,):
        raise ValueError("w_hat must be a 6-element wrench [fx, fy, fz, tx, ty, tz]")

    free_indices = parse_free_axes(free_axes)
    if not free_indices:
        return x_out, R_out

    k_arr = np.broadcast_to(np.asarray(k_adm, dtype=float), (6,))
    b_arr = np.maximum(0.1 * np.abs(k_arr), np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2]))

    v_adm = np.zeros(6, dtype=float)
    for idx in free_indices:
        wi = float(w[idx])
        if abs(wi) > deadband:
            wi_eff = wi - math.copysign(deadband, wi)
            limit = max_lin_vel if idx < 3 else max_ang_vel
            v_adm[idx] = float(np.clip(wi_eff / b_arr[idx], -limit, limit))

    x_out = x_out + v_adm[:3] * float(dt)
    if np.any(v_adm[3:] != 0.0):
        R_out = axis_angle_to_mat(v_adm[3:] * float(dt)) @ R_out
    return x_out, R_out


def dls_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_des: np.ndarray,
    R_des: np.ndarray,
    *,
    damping: float = 0.015,
    gain: float = 0.8,
    max_dq: float = 0.08,
    n_arm: int = 6,
    jacp: np.ndarray | None = None,
    jacr: np.ndarray | None = None,
) -> np.ndarray:
    """Single-step Damped Least Squares IK returning joint-limited target `q_des`."""
    if jacp is None:
        jacp = np.zeros((3, model.nv), dtype=float)
    if jacr is None:
        jacr = np.zeros((3, model.nv), dtype=float)

    x_cur = data.site_xpos[site_id]
    R_cur = data.site_xmat[site_id].reshape(3, 3)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :n_arm], jacr[:, :n_arm]])

    err = np.concatenate([np.asarray(x_des, dtype=float) - x_cur, orientation_error_world(R_des, R_cur)])
    dq = gain * J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
    max_step = float(np.max(np.abs(dq)))
    if max_step > max_dq:
        dq *= max_dq / max_step

    lo = model.jnt_range[:n_arm, 0]
    hi = model.jnt_range[:n_arm, 1]
    return np.clip(data.qpos[:n_arm] + dq, lo, hi)


def pos_admittance_torques(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_des: np.ndarray,
    R_des: np.ndarray,
    *,
    damping: float = 0.015,
    kp_joint: Sequence[float] | np.ndarray = (200.0, 300.0, 200.0, 40.0, 12.0, 2.0),
    kd_joint: Sequence[float] | np.ndarray = (12.0, 20.0, 12.0, 1.8, 0.5, 0.08),
    n_arm: int = 6,
    jacp: np.ndarray | None = None,
    jacr: np.ndarray | None = None,
) -> np.ndarray:
    """Compute joint PD + gravity torques tracking a DLS-IK target for pos_admittance mode."""
    q_des = dls_ik_step(
        model,
        data,
        site_id,
        x_des,
        R_des,
        damping=damping,
        n_arm=n_arm,
        jacp=jacp,
        jacr=jacr,
    )
    kp_j = np.asarray(kp_joint, dtype=float)
    kd_j = np.asarray(kd_joint, dtype=float)
    tau = kp_j * (q_des - data.qpos[:n_arm]) - kd_j * data.qvel[:n_arm] + data.qfrc_bias[:n_arm]
    ctrl_min = model.actuator_ctrlrange[:n_arm, 0]
    ctrl_max = model.actuator_ctrlrange[:n_arm, 1]
    return np.clip(tau, ctrl_min, ctrl_max)


def gripper_control_torques(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mode: Literal["width", "force"],
    *,
    target_width: float = 0.07,
    target_force: float = 8.0,
    width_kp: float = 400.0,
    width_kd: float = 10.0,
    width_max_ctrl: float = 25.0,
    force_kd: float = 1.0,
    k_center: float = 250.0,
    b_center: float = 5.0,
    n_arm: int = 6,
) -> np.ndarray:
    """Compute actuator torques for gripper_joint1 (+) and gripper_joint2 (-)."""
    q_f = data.qpos[n_arm : n_arm + 2]
    qd_f = data.qvel[n_arm : n_arm + 2]
    bias_f = data.qfrc_bias[n_arm : n_arm + 2]

    if mode == "width":
        half_w = 0.5 * float(np.clip(target_width, 0.0, 0.07))
        targets = np.array([half_w, -half_w], dtype=float)
        cmd = np.clip(width_kp * (targets - q_f) - width_kd * qd_f, -width_max_ctrl, width_max_ctrl) + bias_f
    elif mode == "force":
        f = float(np.clip(target_force, 0.0, 25.0))
        # Adaptive centering stiffness: keeps bilateral clamping within +-10% down to 1 N
        kc = min(k_center, 40.0 * f + 60.0)
        center_term = -kc * float(q_f[0] + q_f[1]) - b_center * float(qd_f[0] + qd_f[1])
        cmd = np.array(
            [
                -f - force_kd * float(qd_f[0]) + center_term + float(bias_f[0]),
                +f - force_kd * float(qd_f[1]) + center_term + float(bias_f[1]),
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported gripper mode: {mode!r}")

    ctrl_min = model.actuator_ctrlrange[n_arm : n_arm + 2, 0]
    ctrl_max = model.actuator_ctrlrange[n_arm : n_arm + 2, 1]
    return np.clip(cmd, ctrl_min, ctrl_max)
