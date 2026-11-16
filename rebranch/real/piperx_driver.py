"""Thin offline-importable PiPER-X CAN/SDK driver with lazy piper_sdk import."""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from rebranch.sim.control import Q_HOME


class PiperXHardwareUnavailable(RuntimeError):
    """Raised when real PiPER-X hardware or piper_sdk is unavailable."""


def sdk_joints_to_rad(raw_millideg: Sequence[float]) -> np.ndarray:
    """Convert 6 PiPER SDK joint angles (0.001 deg) to radians."""
    return np.deg2rad(np.asarray(raw_millideg, dtype=float) * 0.001)


def rad_to_sdk_joints(q_rad: Sequence[float]) -> list[int]:
    """Convert 6 joint angles in radians to PiPER SDK integer units (0.001 deg)."""
    return np.rint(np.rad2deg(np.asarray(q_rad, dtype=float)) * 1000.0).astype(int).tolist()


class PiperXDriver:
    """Adapter around `piper_sdk.C_PiperInterface` for real PiPER-X control and feedback."""

    JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
    GRIPPER_MIN_M = 0.0
    GRIPPER_MAX_M = 0.07

    def __init__(
        self,
        can_name: str = "can0",
        *,
        judge_flag: bool = False,
        default_speed_pct: int = 30,
        dry_run: bool = False,
        sdk_client: Any = None,
    ) -> None:
        self.can_name = str(can_name)
        self.judge_flag = bool(judge_flag)
        self.default_speed_pct = int(np.clip(default_speed_pct, 1, 100))
        self.dry_run = bool(dry_run)
        self._sdk = sdk_client
        self._connected = False
        self._dry_q = Q_HOME.copy()
        self._dry_qd = np.zeros(6, dtype=float)
        self._dry_tau = np.zeros(6, dtype=float)
        self._dry_width = 0.07
        self._dry_effort = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, *, piper_init: bool = True) -> None:
        """Connect to PiPER-X CAN interface (imports `piper_sdk` lazily)."""
        if self.dry_run:
            self._connected = True
            return
        if self._sdk is None:
            try:
                from piper_sdk import C_PiperInterface
            except ImportError as exc:
                raise PiperXHardwareUnavailable(
                    "piper_sdk is not installed; use dry_run=True or MujocoEpisode offline"
                ) from exc
            self._sdk = C_PiperInterface(
                can_name=self.can_name,
                judge_flag=self.judge_flag,
                can_auto_init=True,
            )
        self._sdk.ConnectPort(piper_init=piper_init)
        self._connected = True

    def disconnect(self) -> None:
        if self._sdk is not None and not self.dry_run:
            try:
                self._sdk.DisconnectPort()
            except Exception:
                pass
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise PiperXHardwareUnavailable("PiperXDriver is not connected; call connect() first")

    def read_joints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(q_rad[6], qdot_rads[6], tau_nm[6])` from PiPER-X SDK feedback."""
        self._require_connected()
        if self.dry_run:
            return self._dry_q.copy(), self._dry_qd.copy(), self._dry_tau.copy()

        msg = self._sdk.GetArmJointMsgs()
        q_raw = [getattr(msg.joint_state, name) for name in self.JOINT_NAMES]
        q = sdk_joints_to_rad(q_raw)

        # Velocity and motor torque/current from high-speed motor feedback if available
        qd = np.zeros(6, dtype=float)
        tau = np.zeros(6, dtype=float)
        hs_fn = getattr(self._sdk, "GetArmHighSpdInfoMsgs", None)
        if hs_fn is not None:
            hs = hs_fn()
            for i in range(6):
                motor = getattr(hs, f"motor_{i + 1}", None)
                if motor is not None:
                    qd[i] = float(getattr(motor, "motor_speed", 0.0)) * 0.001
                    tau[i] = float(getattr(motor, "effort", 0.0)) * 0.001
        return q, qd, tau

    def read_gripper(self) -> tuple[float, float]:
        """Return `(width_m, effort_n)` from PiPER-X gripper feedback."""
        self._require_connected()
        if self.dry_run:
            return float(self._dry_width), float(self._dry_effort)

        fb = self._sdk.GetArmGripperMsgs()
        width = float(fb.gripper_state.grippers_angle) / 1_000_000.0
        effort = float(getattr(fb.gripper_state, "grippers_effort", 0.0)) / 1000.0
        return float(np.clip(width, self.GRIPPER_MIN_M, self.GRIPPER_MAX_M)), abs(effort)

    def send_joint_positions(self, q_rad: Sequence[float], speed_pct: int | None = None) -> None:
        """Command 6-axis joint target in radians."""
        self._require_connected()
        q = np.asarray(q_rad, dtype=float).reshape(6)
        if not np.isfinite(q).all():
            raise ValueError("Joint targets must be finite")
        if self.dry_run:
            self._dry_q = q.copy()
            return
        spd = self.default_speed_pct if speed_pct is None else int(np.clip(speed_pct, 1, 100))
        self._sdk.MotionCtrl_2(0x01, 0x01, spd, 0x00)
        self._sdk.JointCtrl(*rad_to_sdk_joints(q))

    def send_gripper(self, width_m: float, effort_n: float = 8.0) -> None:
        """Command parallel gripper width (m) and effort (N)."""
        self._require_connected()
        w = float(np.clip(width_m, self.GRIPPER_MIN_M, self.GRIPPER_MAX_M))
        eff = float(np.clip(effort_n, 0.0, 20.0))
        if self.dry_run:
            self._dry_width = w
            self._dry_effort = eff
            return
        self._sdk.GripperCtrl(int(round(w * 1_000_000)), int(round(eff * 1000)), 0x01, 0)

    def stop(self) -> None:
        """Send emergency stop command."""
        if not self._connected or self.dry_run:
            return
        ctrl1 = getattr(self._sdk, "MotionCtrl_1", None)
        if ctrl1 is not None:
            ctrl1(0x01, 0x00, 0x00)

    def sleep_dt(self, dt: float) -> None:
        if not self.dry_run:
            time.sleep(max(0.0, float(dt)))
