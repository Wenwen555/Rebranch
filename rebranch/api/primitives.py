"""Frozen RobotAPI specification and shared API documentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class EpisodeTerminated(Exception):
    """Raised when an episode hits a safety limit, yield/crush limit, or execution budget."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


PRIMITIVES = (
    "get_ee_pose",
    "zero_ee_wrench",
    "get_ee_wrench",
    "get_object_pose",
    "open_gripper",
    "close_gripper",
    "get_gripper_state",
    "move_to",
    "move_until_contact",
    "move_compliant",
)

PHYSCAP_EXTRA_PRIMITIVES = ("get_mass", "get_stiffness")

API_DOC = """\
Robot Policy API (PiPER-X 6-DOF Arm + Parallel Gripper)
=========================================================
Write a single entry-point function `def run(api):` in pure Python.
Rules:
- NO `import` statements are allowed (`math` is already pre-injected).
- Do NOT access private `_`-prefixed attributes or call `eval`/`exec`/`open`/`getattr`.
- Declare tunable numeric/string parameters as top-level uppercase constants with range comments:
    EXAMPLE_FORCE = 6.0  # [1.0, 20.0]
    EXAMPLE_MODE = "a"   # {"a", "b"}
- PiPER-X safe tabletop workspace: x in [0.22, 0.48] m, y in [-0.22, 0.22] m.
- Gripper clamping force range: [1.0, 20.0] N; gripper width range: [0.0, 0.07] m.
- Simulation protocol v2: open/close holds for 0.8 s; motion lasts at least 0.25 s
  and includes 0.2 s endpoint settling (contact detection remains active there).
- Sensor getters return the latest sample without advancing time. Repeated reads at
  one instant are identical samples, not independent noise averaging. A no-op
  move_compliant at the current pose with duration sets an explicit hold interval.
- Motion commands specify setpoints, not guaranteed displacement under contact/load.
  Check actual EE/object poses before interpreting a micro-lift as a hanging load.
- Each policy execution has a hard limit of 200 RobotAPI calls. Every primitive invocation
  counts, including sensor getters; reaching the limit terminates that rollout.

Proprioceptive Sensing Note (Joint-Current Wrench Estimation):
- PiPER-X has NO wrist F/T sensor; `get_ee_wrench()` estimates external wrench from joint
  torques minus gravity/Coriolis compensation (`delta_tau`), so it has lower SNR than a
  dedicated wrist F/T sensor (stiction ~0.1-0.2 Nm, 18 Hz low-pass filter).
- ALWAYS call `api.zero_ee_wrench()` in free space (unloaded) before probing or lifting to
  tare gravity-model residual and thermal/zero-bias drift.
- After `zero_ee_wrench()`, a 300 g hanging payload yields `force[2] approx -2.94 N`.

Primitives:
1. `api.get_ee_pose() -> dict[str, list[float]]`
   Returns `{"pos": [x, y, z], "quat": [w, x, y, z]}` of the TCP in world frame.
2. `api.zero_ee_wrench() -> None`
   Tares the proprioceptive wrench estimator at the current unloaded configuration.
3. `api.get_ee_wrench(frame: str = "world") -> dict[str, list[float]]`
   Returns `{"force": [fx, fy, fz], "torque": [tx, ty, tz], "joint_delta": [d1..d6]}`
   in `"world"` or `"ee"` frame (N, N*m, N*m).
4. `api.get_object_pose(name: str) -> dict[str, list[float]]`
   Returns `{"pos": [x, y, z], "quat": [w, x, y, z]}` for a named scene object.
5. `api.open_gripper(width: float = 0.07) -> None`
   Opens the parallel gripper to `width` in `[0.0, 0.07]` m.
6. `api.close_gripper(force: float = 8.0) -> None`
   Closes the parallel gripper with target clamping force in `[1.0, 20.0]` N.
7. `api.get_gripper_state() -> dict[str, Any]`
   Returns `{"width": float, "effort": float, "grasped": bool}`.
8. `api.move_to(pos, quat=None, speed: float = 0.08) -> None`
   Moves TCP smoothly to `pos=[x,y,z]` (and optional `quat=[w,x,y,z]`) at `speed` (m/s).
9. `api.move_until_contact(pos, quat=None, speed: float = 0.04, max_force: float = 6.0, max_torque: float = 0.8) -> dict[str, Any]`
   Moves toward `pos` and stops early if proprioceptive force/torque exceeds thresholds.
   Returns `{"contact": bool, "pos": [x, y, z], "force": [fx, fy, fz], "torque": [tx, ty, tz]}`.
10. `api.move_compliant(pos, quat=None, free_axes: str = "", speed: float = 0.04, duration: float | None = None) -> None`
   Moves compliantly toward `pos` with low stiffness along `free_axes` (subset of `"xyzXYZ"`,
   lowercase = translation axes, uppercase = rotation axes).
"""


class RobotAPI(ABC):
    """Abstract robot manipulation API shared across toy, MuJoCo, and real backends."""

    @abstractmethod
    def get_ee_pose(self) -> dict[str, list[float]]:
        """Return `{"pos": [x, y, z], "quat": [w, x, y, z]}`."""

    @abstractmethod
    def zero_ee_wrench(self) -> None:
        """Tare proprioceptive wrench bias at the current unloaded configuration."""

    @abstractmethod
    def get_ee_wrench(self, frame: str = "world") -> dict[str, list[float]]:
        """Return `{"force": [fx, fy, fz], "torque": [tx, ty, tz], "joint_delta": [d1..d6]}`."""

    @abstractmethod
    def get_object_pose(self, name: str) -> dict[str, list[float]]:
        """Return `{"pos": [x, y, z], "quat": [w, x, y, z]}` for `name`."""

    @abstractmethod
    def open_gripper(self, width: float = 0.07) -> None:
        """Open gripper to `width` meters (`[0.0, 0.07]`)."""

    @abstractmethod
    def close_gripper(self, force: float = 8.0) -> None:
        """Close gripper with clamping force `force` Newtons (`[1.0, 20.0]`)."""

    @abstractmethod
    def get_gripper_state(self) -> dict[str, Any]:
        """Return `{"width": float, "effort": float, "grasped": bool}`."""

    @abstractmethod
    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.08,
    ) -> None:
        """Move TCP to `pos` (and optional `quat`) at `speed` m/s."""

    @abstractmethod
    def move_until_contact(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.04,
        max_force: float = 6.0,
        max_torque: float = 0.8,
    ) -> dict[str, Any]:
        """Move toward `pos`, stopping on contact; return `{"contact": bool, "pos": [3], "force": [3], "torque": [3]}`."""

    @abstractmethod
    def move_compliant(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        free_axes: str = "",
        speed: float = 0.04,
        duration: float | None = None,
    ) -> None:
        """Move toward `pos` with compliance along `free_axes` (chars in `"xyzXYZ"`)."""
