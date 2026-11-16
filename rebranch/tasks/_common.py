"""Shared helpers, observation scales, and MuJoCo utilities for PiPER-X tasks."""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

from rebranch.sim.control import Q_HOME, mat_to_quat, quat_to_mat
from rebranch.sim.mujoco_episode import MujocoEpisode

TABLE_Z = 0.020
TABLE_POS = (0.35, 0.0, 0.0)
TABLE_SIZE = (0.18, 0.25, 0.020)

DEFAULT_OBS_SCALE: dict[str, float] = {
    "ee_x": 0.004,
    "ee_y": 0.004,
    "ee_z": 0.004,
    "ee_qw": 0.02,
    "ee_qx": 0.02,
    "ee_qy": 0.02,
    "ee_qz": 0.02,
    "fx": 0.4,
    "fy": 0.4,
    "fz": 0.4,
    "tx": 0.06,
    "ty": 0.06,
    "tz": 0.06,
    "jd1": 0.10,
    "jd2": 0.10,
    "jd3": 0.10,
    "jd4": 0.06,
    "jd5": 0.06,
    "jd6": 0.06,
    "grip_width": 0.0015,
    "grip_effort": 0.3,
    "grip_grasped": 0.1,
}

DEFAULT_OBS_API_MAP: dict[str, str] = {
    "ee_x": "api.get_ee_pose()['pos'][0]",
    "ee_y": "api.get_ee_pose()['pos'][1]",
    "ee_z": "api.get_ee_pose()['pos'][2]",
    "ee_qw": "api.get_ee_pose()['quat'][0]",
    "ee_qx": "api.get_ee_pose()['quat'][1]",
    "ee_qy": "api.get_ee_pose()['quat'][2]",
    "ee_qz": "api.get_ee_pose()['quat'][3]",
    "fx": "api.get_ee_wrench()['force'][0]",
    "fy": "api.get_ee_wrench()['force'][1]",
    "fz": "api.get_ee_wrench()['force'][2]",
    "tx": "api.get_ee_wrench()['torque'][0]",
    "ty": "api.get_ee_wrench()['torque'][1]",
    "tz": "api.get_ee_wrench()['torque'][2]",
    "jd1": "api.get_ee_wrench()['joint_delta'][0]",
    "jd2": "api.get_ee_wrench()['joint_delta'][1]",
    "jd3": "api.get_ee_wrench()['joint_delta'][2]",
    "jd4": "api.get_ee_wrench()['joint_delta'][3]",
    "jd5": "api.get_ee_wrench()['joint_delta'][4]",
    "jd6": "api.get_ee_wrench()['joint_delta'][5]",
    "grip_width": "api.get_gripper_state()['width']",
    "grip_effort": "api.get_gripper_state()['effort']",
    "grip_grasped": "api.get_gripper_state()['grasped']",
}


def add_table(
    spec: mujoco.MjSpec,
    *,
    friction: Sequence[float] = (0.7, 0.005, 0.0001),
    pos: Sequence[float] = TABLE_POS,
    size: Sequence[float] = TABLE_SIZE,
    rgba: Sequence[float] = (0.82, 0.80, 0.76, 1.0),
) -> None:
    """Add a static tabletop box geom whose top surface lies at `pos[2] + size[2]`."""
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos),
        size=list(size),
        friction=list(friction),
        rgba=list(rgba),
    )


def geom_pair_normal_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a_id: int,
    geom_b_id: int,
    cf_buf: np.ndarray | None = None,
) -> float:
    """Sum true normal contact forces (N) between two specific geoms."""
    buf = np.zeros(6, dtype=float) if cf_buf is None else cf_buf
    total = 0.0
    pair = (geom_a_id, geom_b_id)
    rev_pair = (geom_b_id, geom_a_id)
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1, g2) == pair or (g1, g2) == rev_pair:
            mujoco.mj_contactForce(model, data, ci, buf)
            total += max(0.0, float(buf[0]))
    return total


def geom_pair_contact_force_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a_id: int,
    geom_b_id: int,
    cf_buf: np.ndarray | None = None,
) -> np.ndarray:
    """Sum 3D contact force vector in world frame acting on `geom_a` from `geom_b`."""
    buf = np.zeros(6, dtype=float) if cf_buf is None else cf_buf
    f_world = np.zeros(3, dtype=float)
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 == geom_a_id and g2 == geom_b_id) or (g1 == geom_b_id and g2 == geom_a_id):
            mujoco.mj_contactForce(model, data, ci, buf)
            frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
            # Contact force in contact frame acts from geom1 onto geom2
            f_on_g2 = frame.T @ buf[:3]
            if g2 == geom_a_id:
                f_world += f_on_g2
            else:
                f_world -= f_on_g2
    return f_world


def body_tilt_rad(data: mujoco.MjData, body_id: int) -> float:
    """Angle in radians between body +Z axis and world +Z axis."""
    z_axis = data.xmat[body_id].reshape(3, 3)[:, 2]
    cos_ang = float(np.clip(z_axis[2], -1.0, 1.0))
    return math.acos(cos_ang)


def gripper_pad_ids(model: mujoco.MjModel) -> tuple[int, int]:
    """Return `(pad_left_id, pad_right_id)`."""
    return model.geom("pad_left").id, model.geom("pad_right").id


def home_tcp_quat() -> list[float]:
    """Return the TCP quaternion [w, x, y, z] at `Q_HOME`."""
    from rebranch.api.proprioception import ProprioceptiveEstimator

    est = ProprioceptiveEstimator()
    _, quat = est.fk(Q_HOME)
    return [float(v) for v in quat]


def rotate_quat_world_z(quat_wxyz: Sequence[float], yaw_rad: float) -> list[float]:
    """Rotate `quat_wxyz` around world +Z axis by `yaw_rad`."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rot = rz @ quat_to_mat(quat_wxyz)
    q = mat_to_quat(rot)
    return [float(v) for v in q]


def clamped_pad_force(
    ep: MujocoEpisode,
    obj_geom_id: int,
    cf_buf: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return `(fn_left, fn_right)` true pad normal forces against `obj_geom_id`."""
    pad_l, pad_r = gripper_pad_ids(ep.model)
    fn_l = geom_pair_normal_force(ep.model, ep.data, pad_l, obj_geom_id, cf_buf=cf_buf)
    fn_r = geom_pair_normal_force(ep.model, ep.data, pad_r, obj_geom_id, cf_buf=cf_buf)
    return fn_l, fn_r
