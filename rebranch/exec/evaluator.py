"""Budget-tracked, optionally multiprocessed policy evaluator."""

from __future__ import annotations

import multiprocessing
from multiprocessing.pool import Pool
from typing import Any, Sequence

from rebranch.exec.sandbox import check_policy_source, run_policy
from rebranch.exec.trace import Trace
from rebranch.sim.backend import TaskBackend


class BudgetExceededError(RuntimeError):
    """Raised when `Evaluator.n_rollouts` would exceed `budget_limit`."""


def _worker_run(
    backend: TaskBackend,
    source: str,
    theta: dict[str, Any],
    seed: int,
    max_lines: int,
    max_calls: int,
    extra_primitives: tuple[str, ...],
) -> Trace:
    episode = backend.make_episode(theta, seed=seed)
    return run_policy(
        source=source,
        episode=episode,
        theta=theta,
        seed=seed,
        max_lines=max_lines,
        max_calls=max_calls,
        extra_primitives=extra_primitives,
    )


class Evaluator:
    """Runs policy rollouts on a `TaskBackend` with rollout budget tracking."""

    def __init__(
        self,
        backend: TaskBackend,
        workers: int = 1,
        max_lines: int = 50000,
        max_calls: int = 200,
        budget_limit: int | None = None,
        extra_primitives: Sequence[str] = (),
    ) -> None:
        self.backend = backend
        self.workers = max(1, int(workers))
        self.max_lines = int(max_lines)
        self.max_calls = int(max_calls)
        self.budget_limit = None if budget_limit is None else int(budget_limit)
        self.extra_primitives = tuple(extra_primitives)
        self.n_rollouts: int = 0
        self._pool: Pool | None = None

    def _consume_budget(self, count: int) -> None:
        if self.budget_limit is not None and self.n_rollouts + count > self.budget_limit:
            raise BudgetExceededError(
                f"Rollout budget {self.budget_limit} exceeded "
                f"(current={self.n_rollouts}, requested={count})"
            )
        self.n_rollouts += count

    def _get_pool(self) -> Pool:
        if self._pool is None:
            ctx = multiprocessing.get_context("spawn")
            self._pool = ctx.Pool(processes=self.workers)
        return self._pool

    def run(self, source: str, theta: dict[str, Any], seed: int = 0) -> Trace:
        """Run a single rollout of `source` on `(theta, seed)`."""
        check_policy_source(source, extra_primitives=self.extra_primitives)
        self._consume_budget(1)
        episode = self.backend.make_episode(theta, seed=seed)
        return run_policy(
            source=source,
            episode=episode,
            theta=theta,
            seed=seed,
            max_lines=self.max_lines,
            max_calls=self.max_calls,
            extra_primitives=self.extra_primitives,
        )

    def run_many(
        self,
        source: str,
        cases: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
        seed: int = 0,
    ) -> list[Trace]:
        """Run `source` over `cases` (list of `theta` dicts or `(theta, seed)` pairs)."""
        if not cases:
            return []
        check_policy_source(source, extra_primitives=self.extra_primitives)

        norm: list[tuple[dict[str, Any], int]] = []
        for idx, item in enumerate(cases):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
                norm.append((item[0], int(item[1])))
            elif isinstance(item, dict):
                norm.append((item, int(seed) + idx))
            else:
                raise TypeError(f"Unsupported case item at index {idx}: {type(item)!r}")

        self._consume_budget(len(norm))
        if self.workers <= 1 or len(norm) == 1:
            return [
                _worker_run(
                    self.backend,
                    source,
                    th,
                    sd,
                    self.max_lines,
                    self.max_calls,
                    self.extra_primitives,
                )
                for th, sd in norm
            ]

        pool = self._get_pool()
        tasks = [
            (
                self.backend,
                source,
                th,
                sd,
                self.max_lines,
                self.max_calls,
                self.extra_primitives,
            )
            for th, sd in norm
        ]
        return pool.starmap(_worker_run, tasks)

    def success_rate(
        self,
        source: str,
        cases: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
        seed: int = 0,
    ) -> float:
        """Evaluate `source` across `cases` and return fraction of successes."""
        if not cases:
            return 0.0
        traces = self.run_many(source, cases, seed=seed)
        return sum(1 for t in traces if t.success) / len(traces)

    def close(self) -> None:
        """Close and join the worker pool if one was spawned."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
