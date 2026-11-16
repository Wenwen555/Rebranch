"""Joint-current proprioceptive wrench estimator using MuJoCo inverse dynamics."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


DEFAULT_PIPERX_XML = (
    Path(__file__).resolve().parents[1] / "tasks" / "assets" / "piperx" / "piperx.xml"
)


class ProprioceptiveEstimator:
    """Estimates 6-axis end-effector wrench from joint-torque residuals."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        tcp_site: str = "tcp",
        damping: float = 0.015,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_PIPERX_XML
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo robot model not found: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.tcp_site = tcp_site
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site
        )
        if self.site_id < 0:
            raise ValueError(f"Site {tcp_site!r} not found in {self.model_path}")
        if self.model.nv < 6 or self.model.nq < 6:
            raise ValueError(
                f"Expected at least 6 DOFs in {self.model_path}, got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.damping = float(damping)
        self.tau_0 = np.zeros(6, dtype=float)
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)

    def _set_arm_state(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        q_arr = np.asarray(q, dtype=float).reshape(6)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:6] = q_arr
        self.data.qvel[:] = 0.0
        if qdot is not None:
            self.data.qvel[:6] = np.asarray(qdot, dtype=float).reshape(6)
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

    def _g_hat(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute expected free-space joint torque (gravity + Coriolis + passive)."""
        self._set_arm_state(q, qdot)
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:6].copy()

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for TCP site: returns `(pos[3], quat_wxyz[4])`."""
        self._set_arm_state(q)
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return 6x6 spatial Jacobian `[J_pos; J_rot]` of TCP site wrt the 6 arm joints."""
        self._set_arm_state(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._jacp.fill(0.0)
        self._jacr.fill(0.0)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, :6], self._jacr[:, :6]]).copy()

    def tare(self, q: np.ndarray, tau_sdk: np.ndarray) -> None:
        """Tare residual joint torque offset at unloaded configuration `q`."""
        q_arr = np.asarray(q, dtype=float).reshape(6)
        tau_arr = np.asarray(tau_sdk, dtype=float).reshape(6)
        self.tau_0 = tau_arr - self._g_hat(q_arr, np.zeros(6, dtype=float))

    def estimate(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        tau_sdk: np.ndarray,
        frame: str = "world",
    ) -> dict[str, list[float]]:
        """Estimate external 6D wrench and per-joint torque residual."""
        if frame not in ("world", "ee"):
            raise ValueError(f"Unknown frame {frame!r}; expected 'world' or 'ee'")
        q_arr = np.asarray(q, dtype=float).reshape(6)
        qdot_arr = np.asarray(qdot, dtype=float).reshape(6)
        tau_arr = np.asarray(tau_sdk, dtype=float).reshape(6)

        delta_tau = (tau_arr - self._g_hat(q_arr, qdot_arr)) - self.tau_0
        J = self.jacobian(q_arr)
        reg = (self.damping ** 2) * np.eye(6, dtype=float)
        # tau_sdk = g_hat - J^T w_ext => delta_tau = -J^T w_ext, so negate damped solve.
        w_world = -J @ np.linalg.solve(J.T @ J + reg, delta_tau)

        if frame == "ee":
            R = self.data.site_xmat[self.site_id].reshape(3, 3)
            force = (R.T @ w_world[:3]).tolist()
            torque = (R.T @ w_world[3:]).tolist()
        else:
            force = w_world[:3].tolist()
            torque = w_world[3:].tolist()

        return {
            "force": [float(v) for v in force],
            "torque": [float(v) for v in torque],
            "joint_delta": [float(v) for v in delta_tau],
        }
