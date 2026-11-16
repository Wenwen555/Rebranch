"""Handwritten privileged capability reference (true theta, not an optimality bound)."""

from __future__ import annotations

import time
from typing import Any

from rebranch.baselines import BaselineResult
from rebranch.exec.sandbox import run_policy
from rebranch.exec.trace import Trace
from rebranch.sim.backend import TaskBackend


class OracleTwinAgent:
    """Privileged runtime evaluator using `backend.oracle_policy(theta)`."""

    def evaluate_episode(
        self,
        backend: TaskBackend,
        theta: dict[str, Any],
        seed: int = 0,
    ) -> tuple[Trace, dict[str, Any]]:
        src = backend.oracle_policy(theta)
        ep = backend.make_episode(theta, seed=seed)
        trace = run_policy(src, ep, theta, seed=seed)
        return trace, {
            "policy_source": src,
            "sim_predicted_success": None,
            "runtime_llm_calls": 0,
            "runtime_tokens": 0,
        }


def run_oracle_twin(backend: TaskBackend) -> BaselineResult:
    """Return the handwritten privileged reference agent (0 dev LLM calls, 0 dev rollouts)."""
    t0 = time.perf_counter()
    nominal_src = backend.oracle_policy(backend.param_space.nominal())
    return BaselineResult(
        method="oracle_twin",
        policy_source=nominal_src,
        n_rollouts=0,
        dev_llm_calls=0,
        dev_prompt_tokens=0,
        dev_completion_tokens=0,
        agent=OracleTwinAgent(),
        program_snapshots=[{"round": 0, "accepted": True, "source": nominal_src}],
        wall_time_s=time.perf_counter() - t0,
        extra={"implementation": "handwritten_privileged_reference", "optimality_certificate": False},
    )
