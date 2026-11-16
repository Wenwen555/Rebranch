"""Execution trace and primitive call record data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecord:
    """Single RobotAPI primitive invocation and post-call SDK observation."""

    index: int
    name: str
    lineno: int
    occurrence: int
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any
    obs: dict[str, float]
    sim_time: float = 0.0


@dataclass
class Trace:
    """Complete execution trace of a policy rollout on `(theta, seed)`."""

    theta: dict[str, Any]
    seed: int
    calls: list[CallRecord] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    success: bool = False
    score: float = -1.0
    failure: str | None = None
    failure_call: int | None = None
    safety_violation: str | None = None
    error: str | None = None
    sim_time: float = 0.0
    wall_time: float = 0.0
    protocol_version: str = "unversioned"
    stdout: str = ""
    stderr: str = ""

    @property
    def path(self) -> frozenset[int]:
        """Set of policy line numbers executed during `run(api)`."""
        return frozenset(self.lines)
