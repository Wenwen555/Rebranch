"""Frozen Episode, EpisodeResult, SNAPSHOT_KEYS, and TaskBackend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rebranch.api.primitives import RobotAPI
from rebranch.sim.params import Continuous, ParamSpace


# Strictly SDK-observable quantities only. Never add ground-truth theta,
# slip_distance, or true contact forces.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qw",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "jd1",
    "jd2",
    "jd3",
    "jd4",
    "jd5",
    "jd6",
    "grip_width",
    "grip_effort",
    "grip_grasped",
)


@dataclass
class EpisodeResult:
    """Outcome of a single policy rollout."""

    success: bool
    score: float
    failure: str | None = None
    safety_violation: str | None = None
    # Task-defined terminal scoring feedback, excluding privileged latent values.
    public_feedback: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"EpisodeResult.score must be in [-1.0, 1.0], got {self.score}")
        if bool(self.success) != (self.score > 0.0):
            raise ValueError(
                f"EpisodeResult invariant violated: success={self.success}, score={self.score}"
            )


class Episode(ABC):
    """Single parameterized rollout instance exposing RobotAPI and SDK snapshot."""

    api: RobotAPI
    sim_time: float = 0.0

    @abstractmethod
    def snapshot(self) -> dict[str, float]:
        """Return a dict containing strictly `SNAPSHOT_KEYS` mapped to floats."""

    @abstractmethod
    def evaluate(self) -> EpisodeResult:
        """Compute final task success, bounded score in [-1, 1], and failure tag."""

    def render_rgb(self, *, width: int = 640, height: int = 480) -> bytes:
        """Return an RGB PNG of the current scene without advancing state."""
        raise NotImplementedError("This episode backend does not provide image rendering")


class TaskBackend(ABC):
    """Abstract manipulation benchmark task parameterized by `ParamSpace`."""

    name: str
    task_description: str
    requires_branching: bool
    param_space: ParamSpace
    obs_scale: dict[str, float]
    obs_api_map: dict[str, str]
    draft_policy: str
    # Public documentation for the task's additional RobotAPI primitives.
    extra_api_doc: str = ""

    @abstractmethod
    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> Episode:
        """Create a fresh episode for parameter vector `theta` and RNG `seed`."""

    def hidden_test_set(self, seed: int = 2026, n: int = 50) -> list[dict[str, Any]]:
        """Sample `n` hidden test instances deterministically from `param_space`."""
        rng = np.random.default_rng(seed)
        return [self.param_space.sample(rng) for _ in range(n)]

    def ood_test_set(self, seed: int = 3026, n: int = 10) -> list[dict[str, Any]]:
        """Sample `n` boundary/extreme out-of-distribution test instances."""
        rng = np.random.default_rng(seed)
        out: list[dict[str, Any]] = []
        for _ in range(n):
            theta = self.param_space.sample(rng)
            for p in self.param_space:
                if isinstance(p, Continuous) and p.name in self.param_space.active:
                    # Bias continuous parameters toward interval edges.
                    if rng.random() < 0.7:
                        edge = p.lo if rng.random() < 0.5 else p.hi
                        jitter = float(rng.uniform(0.0, 0.1 * (p.hi - p.lo)))
                        theta[p.name] = float(
                            min(max(edge + (jitter if edge == p.lo else -jitter), p.lo), p.hi)
                        )
            out.append(theta)
        return out

    @abstractmethod
    def oracle_policy(self, theta: dict[str, Any]) -> str:
        """Return a handwritten capability reference specialized with knowledge of `theta`."""

    @abstractmethod
    def conservative_policy(self) -> str:
        """Return a naive max-force/slow baseline policy (anti-cheat check)."""

    @abstractmethod
    def latent_mode(self, theta: dict[str, Any]) -> str:
        """Return the ground-truth discrete latent regime label for `theta`."""
