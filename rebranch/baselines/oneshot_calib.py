"""1-Shot Sim-Calibration baseline: fixed probe -> NN twin match -> Single-Twin CaP."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from rebranch.baselines import (
    BaselineResult,
    build_safe_probe_source,
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


class OneShotCalibAgent:
    """Runtime 1-Shot Sim-Calibration agent (probe -> NN twin -> Single-Twin CaP)."""

    def __init__(
        self,
        probe_source: str,
        twin_bank: list[dict[str, Any]],
        probe_obs_bank: list[dict[str, float]],
        calibrated_policies: list[str],
        calibrated_pred_ok: list[bool],
    ) -> None:
        self.probe_source = probe_source
        self.twin_bank = twin_bank
        self.probe_obs_bank = probe_obs_bank
        self.calibrated_policies = calibrated_policies
        self.calibrated_pred_ok = calibrated_pred_ok

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
        best_idx = 0
        best_d2 = float("inf")
        for idx, obs_j in enumerate(self.probe_obs_bank):
            d2 = obs_distance_sq(obs_real, obs_j, backend.obs_scale)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx

        chosen_src = self.calibrated_policies[best_idx]
        pred_ok = self.calibrated_pred_ok[best_idx]

        remaining = 200 - len(t_probe.calls)
        if remaining <= 0:
            trace = finalize_episode_trace(
                ep, t_probe, execution_error="max_calls (200) exhausted before action phase"
            )
        else:
            t_act = run_policy(
                chosen_src, ep, theta, seed=seed + 10000,
                max_calls=remaining, evaluate=False,
            )
            trace = merge_two_phase_traces(ep, t_probe, t_act, max_calls=200)
        trace.wall_time = time.perf_counter() - t0
        return trace, {
            "policy_source": chosen_src,
            "matched_twin_index": best_idx,
            "probe_features": obs_real,
            "sim_predicted_success": pred_ok,
            "runtime_llm_calls": 0,
            "runtime_tokens": 0,
            "wall_time_s": trace.wall_time,
        }


def _select_diverse_indices(
    probe_obs_bank: list[dict[str, float]],
    obs_scale: dict[str, float],
    k: int,
) -> list[int]:
    """Farthest-point sampling over SDK probe observations to pick prototype twins."""
    if not probe_obs_bank or k <= 0:
        return []
    selected = [0]
    while len(selected) < min(k, len(probe_obs_bank)):
        best_i = -1
        best_min_d = -1.0
        for i, obs_i in enumerate(probe_obs_bank):
            if i in selected:
                continue
            min_d = min(
                obs_distance_sq(obs_i, probe_obs_bank[j], obs_scale) for j in selected
            )
            if min_d > best_min_d:
                best_min_d = min_d
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
    return selected


def run_oneshot_calib(
    backend: TaskBackend,
    llm: LLMClient,
    *,
    seed: int = 0,
    workers: int = 1,
    max_rollouts: int = 300,
    max_llm_calls: int = 8,
    n_twins: int = 24,
) -> BaselineResult:
    """1-Shot Sim-Calibration: fixed probe bank + Single-Twin CaP on diverse prototypes."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    probe_src = build_safe_probe_source(backend)
    budget = max(4, int(max_rollouts))
    llm_budget = max(0, int(max_llm_calls))

    m_twins = max(4, min(int(n_twins), budget // 4))
    twin_bank = [backend.param_space.sample(rng) for _ in range(m_twins)]
    probe_obs_bank: list[dict[str, float]] = []

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

        n_protos = min(max(1, llm_budget), 4, len(twin_bank))
        proto_indices = _select_diverse_indices(
            probe_obs_bank, backend.obs_scale, n_protos
        )

        proto_policies: dict[int, tuple[str, bool]] = {}
        for p_idx in proto_indices:
            if ev.n_rollouts >= budget:
                proto_policies[p_idx] = (backend.draft_policy, False)
                continue
            th = twin_bank[p_idx]
            try:
                tr0 = ev.run(backend.draft_policy, th, seed=seed + p_idx)
            except BudgetExceededError:
                proto_policies[p_idx] = (backend.draft_policy, False)
                continue

            best_src = backend.draft_policy
            best_tr = tr0
            if not best_tr.success and dev_llm_calls < llm_budget:
                prompt = build_single_twin_prompt(
                    task_description=backend.task_description,
                    current_code=best_src,
                    failure_summary=summarize_trace_for_llm(best_tr),
                )
                resp = llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.2)
                dev_llm_calls += 1
                dev_pt += int(resp.prompt_tokens)
                dev_ct += int(resp.completion_tokens)
                cand = extract_python_code(resp.text)
                try:
                    check_policy_source(cand)
                    if ev.n_rollouts < budget:
                        cand_tr = ev.run(cand, th, seed=seed + p_idx)
                        if cand_tr.success or cand_tr.score > best_tr.score:
                            best_src = cand
                            best_tr = cand_tr
                            program_snapshots.append(
                                {"round": dev_llm_calls, "accepted": True, "source": best_src}
                            )
                except (StaticCheckError, BudgetExceededError) as exc:
                    round_logs.append({"proto": p_idx, "error": str(exc)})

            # Optional quick constant tuning on prototype twin if still failing
            rem = budget - ev.n_rollouts
            if not best_tr.success and rem >= 8:
                tuned_src, used_r, _ = tune_constants_cem(
                    backend, best_src, [th], seed=seed + 300 + p_idx, max_rollouts=min(12, rem)
                )
                ev.n_rollouts += used_r
                if tuned_src != best_src and ev.n_rollouts < budget:
                    try:
                        t_tr = ev.run(tuned_src, th, seed=seed + p_idx)
                        if t_tr.success or t_tr.score > best_tr.score:
                            best_src = tuned_src
                            best_tr = t_tr
                    except BudgetExceededError:
                        pass

            proto_policies[p_idx] = (best_src, bool(best_tr.success))

        # Assign every twin in bank to its nearest calibrated prototype in SDK probe space
        calibrated_policies: list[str] = []
        calibrated_pred_ok: list[bool] = []
        for idx, obs_i in enumerate(probe_obs_bank):
            if idx in proto_policies:
                src, ok = proto_policies[idx]
            elif proto_indices:
                nearest_p = min(
                    proto_indices,
                    key=lambda p: obs_distance_sq(
                        obs_i, probe_obs_bank[p], backend.obs_scale
                    ),
                )
                src, ok = proto_policies[nearest_p]
            else:
                src, ok = backend.draft_policy, False
            calibrated_policies.append(src)
            calibrated_pred_ok.append(ok)

        n_rollouts = ev.n_rollouts

    agent = OneShotCalibAgent(
        probe_source=probe_src,
        twin_bank=twin_bank,
        probe_obs_bank=probe_obs_bank,
        calibrated_policies=calibrated_policies,
        calibrated_pred_ok=calibrated_pred_ok,
    )
    return BaselineResult(
        method="oneshot_calib",
        policy_source=calibrated_policies[0] if calibrated_policies else backend.draft_policy,
        n_rollouts=n_rollouts,
        dev_llm_calls=dev_llm_calls,
        dev_prompt_tokens=dev_pt,
        dev_completion_tokens=dev_ct,
        agent=agent,
        round_logs=round_logs,
        program_snapshots=program_snapshots,
        wall_time_s=time.perf_counter() - t0,
        extra={"n_twins": len(twin_bank), "n_prototypes": len(proto_indices)},
    )
