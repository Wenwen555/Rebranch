"""Local candidate-program selection proxy, legacy key `twin_mpc`. No receding-horizon replanning."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from rebranch.baselines import (
    BaselineResult,
    build_safe_probe_source,
    combine_two_phase_episode,
    finalize_episode_trace,
    merge_two_phase_traces,
    obs_distance_sq,
    probe_features,
    tune_constants_cem,
)
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.exec.evaluator import BudgetExceededError, Evaluator
from rebranch.exec.sandbox import StaticCheckError, check_policy_source, run_policy
from rebranch.exec.trace import Trace
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.llm.prompts import SYSTEM_PROMPT, build_single_twin_prompt
from rebranch.sim.backend import TaskBackend


class TwinMPCAgent:
    """Runtime agent matching probe SDK readings against a particle bank of twins."""

    def __init__(
        self,
        probe_source: str,
        twin_bank: list[dict[str, Any]],
        probe_obs_bank: list[dict[str, float]],
        candidate_policies: list[str],
        *,
        top_k_twins: int = 4,
        runtime_sim_budget_per_episode: int = 24,
    ) -> None:
        self.probe_source = probe_source
        self.twin_bank = twin_bank
        self.probe_obs_bank = probe_obs_bank
        self.candidate_policies = candidate_policies
        self.top_k_twins = max(1, int(top_k_twins))
        self.runtime_sim_budget_per_episode = max(0, int(runtime_sim_budget_per_episode))

    def evaluate_episode(
        self,
        backend: TaskBackend,
        theta: dict[str, Any],
        seed: int = 0,
    ) -> tuple[Trace, dict[str, Any]]:
        t0 = time.perf_counter()
        ep = backend.make_episode(theta, seed=seed)
        t_probe = run_policy(
            self.probe_source, ep, theta, seed=seed, max_calls=200, evaluate=False
        )
        if t_probe.error is not None or not t_probe.calls:
            trace = finalize_episode_trace(
                ep, t_probe,
                execution_error=("probe produced no SDK calls" if not t_probe.calls and t_probe.error is None else None),
            )
            return trace, {
                "policy_source": self.probe_source,
                "sim_predicted_success": False,
                "runtime_llm_calls": 0,
                "runtime_tokens": 0,
                "wall_time_s": time.perf_counter() - t0,
            }

        obs_real = probe_features(t_probe, self.probe_source)
        scored_twins: list[tuple[float, dict[str, Any]]] = []
        for th_j, obs_j in zip(self.twin_bank, self.probe_obs_bank):
            d2 = obs_distance_sq(obs_real, obs_j, backend.obs_scale)
            w = math.exp(-0.5 * min(d2, 50.0))
            scored_twins.append((w, th_j))

        scored_twins.sort(key=lambda x: x[0], reverse=True)
        top_twins = scored_twins[: self.top_k_twins]
        w_sum = sum(w for w, _ in top_twins) or 1.0
        norm_twins = [(w / w_sum, th) for w, th in top_twins]

        best_src = self.candidate_policies[0]
        best_val = -1e9
        best_pred_sr = None
        runtime_sim_rollouts = 0
        runtime_sim_time = 0.0
        for cand_src in self.candidate_policies:
            if runtime_sim_rollouts + len(norm_twins) > self.runtime_sim_budget_per_episode:
                break
            exp_sr = 0.0
            exp_score = 0.0
            for idx, (w, th_j) in enumerate(norm_twins):
                sim_ep = backend.make_episode(th_j, seed=seed + 500 + idx)
                sim_tr = combine_two_phase_episode(
                    sim_ep, self.probe_source, cand_src, th_j, seed=seed + 500 + idx
                )
                runtime_sim_rollouts += 1
                runtime_sim_time += sim_tr.sim_time
                exp_sr += w * (1.0 if sim_tr.success else 0.0)
                exp_score += w * float(sim_tr.score)
            val = exp_sr + 0.25 * exp_score
            if val > best_val:
                best_val = val
                best_src = cand_src
                best_pred_sr = exp_sr

        remaining = 200 - len(t_probe.calls)
        if remaining <= 0:
            trace = finalize_episode_trace(
                ep, t_probe, execution_error="max_calls (200) exhausted before action phase"
            )
        else:
            t_act = run_policy(
                best_src, ep, theta, seed=seed + 10000,
                max_calls=remaining, evaluate=False,
            )
            trace = merge_two_phase_traces(ep, t_probe, t_act, max_calls=200)
        trace.wall_time = time.perf_counter() - t0
        return trace, {
            "policy_source": best_src,
            "probe_features": obs_real,
            "implementation": "candidate_program_selection_proxy",
            "sim_predicted_success": None if best_pred_sr is None else best_pred_sr >= 0.5,
            "runtime_sim_rollouts": runtime_sim_rollouts,
            "runtime_sim_time_s": runtime_sim_time,
            "runtime_sim_budget_per_episode": self.runtime_sim_budget_per_episode,
            "runtime_sim_stop": "budget_exhausted" if runtime_sim_rollouts < len(self.candidate_policies)*len(norm_twins) else "all_candidates_evaluated",
            "runtime_llm_calls": 0,
            "runtime_tokens": 0,
            "wall_time_s": trace.wall_time,
        }


def run_twin_mpc(
    backend: TaskBackend,
    llm: LLMClient,
    *,
    seed: int = 0,
    workers: int = 1,
    max_rollouts: int = 300,
    max_llm_calls: int = 8,
    n_twins: int = 16,
    runtime_sim_budget_per_episode: int = 24,
    receding_horizon: bool = False,
) -> BaselineResult:
    """Build Twin-MPC particle bank & candidate policies under matched dev budget."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    probe_src = build_safe_probe_source(backend)
    budget = max(4, int(max_rollouts))
    llm_budget = max(0, int(max_llm_calls))

    m_twins = max(4, min(int(n_twins), budget // 4))
    twin_bank = [backend.param_space.sample(rng) for _ in range(m_twins)]
    probe_obs_bank: list[dict[str, float]] = []
    candidates: list[str] = [backend.draft_policy]

    dev_llm_calls = 0
    dev_pt = 0
    dev_ct = 0
    round_logs: list[dict[str, Any]] = []
    program_snapshots: list[dict[str, Any]] = [
        {"round": 0, "accepted": True, "source": backend.draft_policy}
    ]

    with Evaluator(backend, workers=workers, budget_limit=budget) as ev:
        probe_traces = ev.run_many(probe_src, twin_bank, seed=seed)
        for tr in probe_traces:
            probe_obs_bank.append(probe_features(tr, probe_src))

        # Evaluate draft on twin bank to find failure clusters and build candidate policies
        eval_n = min(len(twin_bank), max(1, (budget - ev.n_rollouts) // 3))
        try:
            draft_traces = ev.run_many(backend.draft_policy, twin_bank[:eval_n], seed=seed + 10)
        except BudgetExceededError:
            draft_traces = []

        failing = [tr for tr in draft_traces if not tr.success]
        n_repairs = min(llm_budget, 3, len(failing))
        for r in range(n_repairs):
            if ev.n_rollouts >= budget:
                break
            fail_tr = failing[r]
            task_description = backend.task_description
            if receding_horizon:
                task_description += ("\n\nRolling MPC contract: at runtime this program is resumed after each "
                    "selected RobotAPI action. The executor reruns the full candidate against "
                    "fresh SDK observations, skips this candidate's already executed actuator-call prefix, and "
                    "executes only its next actuator call. Candidate forecasts are evaluated "
                    "for a finite H-action horizon, but only the first action is executed before replanning.")
            prompt = build_single_twin_prompt(
                task_description=task_description,
                current_code=candidates[-1],
                failure_summary=summarize_trace_for_llm(fail_tr),
            )
            resp = llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.2)
            dev_llm_calls += 1
            dev_pt += int(resp.prompt_tokens)
            dev_ct += int(resp.completion_tokens)
            cand_src = extract_python_code(resp.text)
            try:
                check_policy_source(cand_src)
                if cand_src not in candidates:
                    candidates.append(cand_src)
                    program_snapshots.append(
                        {"round": r + 1, "accepted": True, "source": cand_src}
                    )
            except StaticCheckError as exc:
                round_logs.append({"round": r + 1, "error": str(exc)})

        # Also generate up to 2 constant-tuned candidate variants within remaining budget
        rem = budget - ev.n_rollouts
        if rem >= 12 and failing:
            tuned_src, used_r, _ = tune_constants_cem(
                backend,
                backend.draft_policy,
                [failing[0].theta],
                seed=seed + 200,
                max_rollouts=min(16, rem),
            )
            ev.n_rollouts += used_r
            if tuned_src not in candidates:
                candidates.append(tuned_src)

        n_rollouts = ev.n_rollouts

    agent = TwinMPCAgent(
        probe_source=probe_src,
        twin_bank=twin_bank,
        probe_obs_bank=probe_obs_bank,
        candidate_policies=candidates[:6],
        runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
    )
    return BaselineResult(
        method="twin_mpc",
        policy_source=candidates[0],
        n_rollouts=n_rollouts,
        dev_llm_calls=dev_llm_calls,
        dev_prompt_tokens=dev_pt,
        dev_completion_tokens=dev_ct,
        agent=agent,
        round_logs=round_logs,
        program_snapshots=program_snapshots,
        wall_time_s=time.perf_counter() - t0,
        extra={"n_twins": len(twin_bank), "n_candidates": len(candidates[:6]), "implementation": "candidate_program_selection_proxy", "receding_horizon": False, "runtime_sim_budget_per_episode": runtime_sim_budget_per_episode},
    )
