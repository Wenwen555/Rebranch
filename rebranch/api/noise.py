"""Realistic PiPER-X joint-torque and gripper measurement noise model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CALIB_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "piperx_calib.yaml"
)


class JointNoiseModel:
    """Applies calibrated stiction, viscous friction, drift, quantization, noise, and 18 Hz LPF."""

    def __init__(
        self,
        calib_path: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.calib_path = Path(calib_path) if calib_path is not None else DEFAULT_CALIB_PATH
        self.enabled = bool(enabled)
        cfg: dict[str, Any] = yaml.safe_load(self.calib_path.read_text())

        arm = cfg["arm"]
        self.coulomb_nm = np.asarray(arm["coulomb_nm"], dtype=float).reshape(6)
        self.stiction_vel_eps = float(arm.get("stiction_vel_eps", 0.02))
        self.viscous_nms = np.asarray(arm["viscous_nms"], dtype=float).reshape(6)
        self.bias_drift_std_nm = np.asarray(arm["bias_drift_std_nm"], dtype=float).reshape(6)
        self.gain_error_std = np.asarray(arm["gain_error_std"], dtype=float).reshape(6)
        self.torque_quant_nm = np.asarray(arm["torque_quant_nm"], dtype=float).reshape(6)
        self.noise_std_nm = np.asarray(arm["noise_std_nm"], dtype=float).reshape(6)
        self.lpf_cutoff_hz = float(arm.get("lpf_cutoff_hz", 18.0))
        self.dt = float(arm.get("dt", 0.002))
        self._lpf_alpha = 1.0 - math.exp(-2.0 * math.pi * self.lpf_cutoff_hz * self.dt)

        grip = cfg["gripper"]
        self.width_noise_std_m = float(grip.get("width_noise_std_m", 0.0005))
        self.width_quant_m = float(grip.get("width_quant_m", 0.0002))
        self.width_min_m = float(grip.get("width_min_m", 0.0))
        self.width_max_m = float(grip.get("width_max_m", 0.07))
        self.effort_noise_std_n = float(grip.get("effort_noise_std_n", 0.15))
        self.effort_quant_n = float(grip.get("effort_quant_n", 0.05))
        self.effort_bias_std_n = float(grip.get("effort_bias_std_n", 0.10))

        self.arm_bias = np.zeros(6, dtype=float)
        self.arm_gain = np.ones(6, dtype=float)
        self.lpf_state: np.ndarray | None = None
        self.gripper_effort_bias = 0.0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset episode-level bias drift, gain error, and low-pass filter state."""
        self.lpf_state = None
        if not self.enabled or rng is None:
            self.arm_bias = np.zeros(6, dtype=float)
            self.arm_gain = np.ones(6, dtype=float)
            self.gripper_effort_bias = 0.0
            return
        self.arm_bias = rng.normal(0.0, self.bias_drift_std_nm)
        self.arm_gain = 1.0 + rng.normal(0.0, self.gain_error_std)
        self.gripper_effort_bias = float(rng.normal(0.0, self.effort_bias_std_n))

    def apply_arm(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        tau_true: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return noisy, quantized, low-pass-filtered arm torque measurement `(6,)`."""
        tau_arr = np.asarray(tau_true, dtype=float).reshape(6)
        if not self.enabled:
            return tau_arr.copy()

        _ = np.asarray(q, dtype=float).reshape(6)
        qdot_arr = np.asarray(qdot, dtype=float).reshape(6)

        stiction = self.coulomb_nm * np.tanh(qdot_arr / max(self.stiction_vel_eps, 1e-6))
        viscous = self.viscous_nms * qdot_arr
        tau_raw = self.arm_gain * tau_arr + stiction + viscous + self.arm_bias
        tau_noisy = tau_raw + rng.normal(0.0, self.noise_std_nm)
        tau_quant = np.where(
            self.torque_quant_nm > 0.0,
            np.round(tau_noisy / self.torque_quant_nm) * self.torque_quant_nm,
            tau_noisy,
        )

        if self.lpf_state is None:
            self.lpf_state = tau_quant.copy()
        else:
            self.lpf_state = (
                self._lpf_alpha * tau_quant + (1.0 - self._lpf_alpha) * self.lpf_state
            )
        return self.lpf_state.copy()

    def apply_gripper(
        self,
        width_true: float,
        effort_true: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Return noisy, quantized `(width_meas, effort_meas)`."""
        if not self.enabled:
            return (float(width_true), float(effort_true))

        w = float(width_true) + float(rng.normal(0.0, self.width_noise_std_m))
        if self.width_quant_m > 0.0:
            w = round(w / self.width_quant_m) * self.width_quant_m
        w = min(max(w, self.width_min_m), self.width_max_m)

        eff = (
            float(effort_true)
            + self.gripper_effort_bias
            + float(rng.normal(0.0, self.effort_noise_std_n))
        )
        if self.effort_quant_n > 0.0:
            eff = round(eff / self.effort_quant_n) * self.effort_quant_n
        return (float(w), float(eff))
