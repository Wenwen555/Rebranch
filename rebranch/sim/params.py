"""Continuous and discrete task parameter spaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Continuous:
    """Bounded continuous physical or nuisance parameter."""

    name: str
    lo: float
    hi: float
    nominal: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"{self.name}: lo={self.lo} > hi={self.hi}")
        if not (self.lo <= self.nominal <= self.hi):
            raise ValueError(
                f"{self.name}: nominal={self.nominal} not in [{self.lo}, {self.hi}]"
            )


@dataclass(frozen=True)
class Discrete:
    """Finite discrete physical or nuisance parameter."""

    name: str
    choices: tuple[Any, ...]
    nominal: Any

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.name}: choices cannot be empty")
        if self.nominal not in self.choices:
            raise ValueError(
                f"{self.name}: nominal={self.nominal!r} not in {self.choices!r}"
            )


class ParamSpace:
    """Mixed continuous/discrete parameter space with active-subset masking."""

    def __init__(
        self,
        params: Sequence[Continuous | Discrete],
        active: set[str] | None = None,
    ) -> None:
        self.params: tuple[Continuous | Discrete, ...] = tuple(params)
        self._by_name: dict[str, Continuous | Discrete] = {}
        for p in self.params:
            if p.name in self._by_name:
                raise ValueError(f"Duplicate parameter name: {p.name!r}")
            self._by_name[p.name] = p

        if active is None:
            self.active: set[str] = set(self._by_name)
        else:
            unknown = set(active) - set(self._by_name)
            if unknown:
                raise ValueError(f"Unknown active parameter(s): {sorted(unknown)}")
            self.active = set(active)

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Continuous | Discrete:
        return self._by_name[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def nominal(self) -> dict[str, Any]:
        """Return a fresh nominal parameter dictionary."""
        return {p.name: p.nominal for p in self.params}

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample active parameters uniformly; inactive parameters stay nominal."""
        out: dict[str, Any] = {}
        for p in self.params:
            if p.name not in self.active:
                out[p.name] = p.nominal
            elif isinstance(p, Continuous):
                out[p.name] = float(rng.uniform(p.lo, p.hi))
            else:
                idx = int(rng.integers(0, len(p.choices)))
                out[p.name] = p.choices[idx]
        return out

    def with_active(self, names: Iterable[str]) -> ParamSpace:
        """Return a view of this space with only `names` active."""
        return ParamSpace(self.params, active=set(names))

    def interpolate(
        self, a: dict[str, Any], b: dict[str, Any], t: float
    ) -> dict[str, Any]:
        """Interpolate between `a` and `b` at `t in [0, 1]`."""
        t_clamped = min(max(float(t), 0.0), 1.0)
        out: dict[str, Any] = {}
        for p in self.params:
            if p.name not in self.active:
                out[p.name] = a[p.name]
            elif isinstance(p, Continuous):
                val = (1.0 - t_clamped) * float(a[p.name]) + t_clamped * float(b[p.name])
                out[p.name] = float(min(max(val, p.lo), p.hi))
            else:
                out[p.name] = a[p.name] if t_clamped < 0.5 else b[p.name]
        return out

    def distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """Normalized Euclidean distance over active parameters."""
        sq_sum = 0.0
        for p in self.params:
            if p.name not in self.active:
                continue
            if isinstance(p, Continuous):
                span = max(p.hi - p.lo, 1e-12)
                diff = (float(a[p.name]) - float(b[p.name])) / span
                sq_sum += diff * diff
            else:
                sq_sum += 0.0 if a[p.name] == b[p.name] else 1.0
        return math.sqrt(sq_sum)
