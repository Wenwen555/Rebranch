"""Custom particle-twin MPC with public action-prefix replay and receding decisions."""
from __future__ import annotations

import math
import time
from typing import Any

from rebranch.baselines import BaselineResult, build_safe_probe_source, finalize_episode_trace, probe_features
from rebranch.baselines.twin_mpc import run_twin_mpc
from rebranch.exec.sandbox import run_policy
from rebranch.exec.trace import CallRecord, Trace
from rebranch.llm.client import LLMClient
from rebranch.sim.backend import Episode, TaskBackend


_SENSORY = {"get_ee_pose", "get_ee_wrench", "get_object_pose", "get_gripper_state"}


def _prefix_record(call: CallRecord) -> dict[str, Any]:
    return {"name": call.name, "args": tuple(call.args), "kwargs": dict(call.kwargs)}


def _replay_prefix(ep: Episode, prefix: list[dict[str, Any]], replay_count: list[int]) -> None:
    """Replay public RobotAPI calls; count each completed invocation even on later error."""
    for record in prefix:
        getattr(ep.api, record["name"])(*record["args"], **record["kwargs"])
        replay_count[0] += 1


def _combine_final(ep: Episode, traces: list[Trace], theta: dict[str, Any], seed: int,
                   error: str | None = None) -> Trace:
    calls: list[CallRecord] = []
    lines: list[int] = []
    wall = 0.0
    safety = None
    for trace in traces:
        offset = len(calls)
        for call in trace.calls:
            call.index = offset + len(calls) - offset
            calls.append(call)
        lines.extend(trace.lines)
        wall += trace.wall_time
        safety = safety or trace.safety_violation
        error = error or trace.error
    for i, call in enumerate(calls):
        call.index = i
    merged = Trace(theta=dict(theta), seed=seed, calls=calls, lines=lines,
                   error=error, failure=error, safety_violation=safety,
                   sim_time=float(getattr(ep, "sim_time", 0.0)), wall_time=wall,
                   protocol_version=getattr(ep, "protocol_version", "unversioned"))
    return finalize_episode_trace(ep, merged, execution_error=error)


class TwinMPCRecedingAgent:
    """At each actual API action, replay public history into sampled twins and replan."""
    def __init__(self, probe_source: str, twin_bank: list[dict[str, Any]],
                 probe_obs_bank: list[dict[str, float]], candidate_policies: list[str], *,
                 prediction_horizon: int = 3, top_k_twins: int = 2,
                 runtime_sim_budget_per_episode: int = 24, max_calls: int = 200):
        if int(prediction_horizon) < 2:
            raise ValueError("Twin MPC prediction_horizon must be at least 2")
        self.probe_source = probe_source
        self.twin_bank = twin_bank
        self.probe_obs_bank = probe_obs_bank
        self.candidate_policies = candidate_policies
        self.prediction_horizon = int(prediction_horizon)
        self.top_k_twins = max(1, int(top_k_twins))
        self.runtime_sim_budget_per_episode = max(0, int(runtime_sim_budget_per_episode))
        self.max_calls = int(max_calls)

    def evaluate_episode(self, backend: TaskBackend, theta: dict[str, Any], seed: int = 0):
        started = time.perf_counter()
        ep = backend.make_episode(theta, seed=seed)
        actual_traces: list[Trace] = []
        probe_trace = run_policy(self.probe_source, ep, theta, seed=seed,
                                 max_calls=self.max_calls, evaluate=False)
        actual_traces.append(probe_trace)
        runtime_sim_rollouts = runtime_replay_calls = 0
        runtime_sim_time = 0.0
        plans: list[dict[str, Any]] = []
        if probe_trace.error or not probe_trace.calls:
            err = probe_trace.error or "probe produced no SDK calls"
            final = _combine_final(ep, actual_traces, theta, seed, error=err)
            return final, {"policy_source": self.probe_source, "sim_predicted_success": None,
                           "runtime_sim_rollouts": 0, "runtime_replay_api_calls": 0,
                           "runtime_sim_time_s": 0.0, "planning_steps": [],
                           "wall_time_s": time.perf_counter()-started}

        probe_obs = probe_features(probe_trace, self.probe_source)
        distances = []
        for idx, (tw, obs) in enumerate(zip(self.twin_bank, self.probe_obs_bank)):
            d2 = sum(((probe_obs[k] - obs[k]) / max(backend.obs_scale.get(k, 1.0), 1e-6)) ** 2
                     for k in probe_obs if k in obs)
            distances.append((float(d2), idx, tw))
        selected = sorted(distances, key=lambda x: x[0])[:self.top_k_twins]
        prefix = [_prefix_record(c) for c in probe_trace.calls]
        candidate_action_counts = [0 for _ in self.candidate_policies]
        candidate_action_results: list[list[Any]] = [[] for _ in self.candidate_policies]
        last_predicted_success: float | None = None
        stop_reason = "runtime_sim_budget_exhausted"
        decision_index = 0

        while len([c for tr in actual_traces for c in tr.calls]) < self.max_calls:
            # Full candidate × particle comparison is one MPC decision. Do not rank
            # from an incomplete subset when the predeclared prediction budget is short.
            needed = len(self.candidate_policies) * len(selected)
            if not self.candidate_policies or runtime_sim_rollouts + needed > self.runtime_sim_budget_per_episode:
                stop_reason = "runtime_sim_budget_exhausted"
                break
            current_obs = ep.snapshot()
            scored: list[tuple[float, str, float, list[dict[str, Any]]]] = []
            for cand_index, candidate in enumerate(self.candidate_policies):
                weighted_score = weighted_success = weight_sum = 0.0
                predictions = []
                for particle_index, (_, bank_index, twin_theta) in enumerate(selected):
                    sim_seed = seed + 50000 + bank_index
                    sim_ep = backend.make_episode(twin_theta, seed=sim_seed)
                    before = float(getattr(sim_ep, "sim_time", 0.0))
                    replay_count = [0]
                    try:
                        _replay_prefix(sim_ep, prefix, replay_count)
                        runtime_replay_calls += replay_count[0]
                        weight_dist = sum(((current_obs[k] - sim_ep.snapshot().get(k, current_obs[k])) /
                                           max(backend.obs_scale.get(k, 1.0), 1e-6)) ** 2
                                          for k in current_obs)
                        weight = math.exp(-0.5 * min(weight_dist, 50.0))
                        source = candidate
                        remaining = max(1, self.max_calls - len(prefix))
                        pred = run_policy(source, sim_ep, twin_theta, seed=sim_seed,
                                          max_calls=remaining, planning_horizon_actions=self.prediction_horizon,
                                          skip_actions=candidate_action_counts[cand_index],
                                          skipped_action_results=candidate_action_results[cand_index])
                        runtime_sim_rollouts += 1
                        runtime_sim_time += float(getattr(sim_ep, "sim_time", 0.0)) - before
                        value = 1.0 if pred.success else float(pred.score)
                        weighted_score += weight * value
                        weighted_success += weight * (1.0 if pred.success else 0.0)
                        weight_sum += weight
                        predictions.append({"particle_index": bank_index, "weight": weight,
                                            "progress_score": value, "success": pred.success,
                                            "forecast_actions": sum(c.name not in _SENSORY for c in pred.calls),
                                            "error": pred.error})
                    except Exception as exc:
                        runtime_replay_calls += replay_count[0]
                        runtime_sim_rollouts += 1
                        runtime_sim_time += float(getattr(sim_ep, "sim_time", 0.0)) - before
                        predictions.append({"particle_index": bank_index, "weight": 0.0,
                                            "progress_score": -1.0, "success": False,
                                            "error": f"{type(exc).__name__}: {exc}"})
                if weight_sum:
                    scored.append((weighted_score / weight_sum, candidate,
                                   weighted_success / weight_sum, predictions))
            if not scored:
                stop_reason = "no_valid_twin_forecast"
                break
            scored.sort(key=lambda item: item[0], reverse=True)
            value, selected_source, pred_success, predictions = scored[0]
            selected_index = self.candidate_policies.index(selected_source)
            last_predicted_success = pred_success
            remaining_actual = self.max_calls - sum(len(tr.calls) for tr in actual_traces)
            if remaining_actual <= 0:
                stop_reason = "actual_api_budget_exhausted"
                break
            actual_step = run_policy(selected_source, ep, theta, seed=seed + 10000 + decision_index,
                                     max_calls=remaining_actual, evaluate=False,
                                     planning_horizon_actions=1,
                                     skip_actions=candidate_action_counts[selected_index],
                                     skipped_action_results=candidate_action_results[selected_index])
            actual_traces.append(actual_step)
            actual_actions = [c for c in actual_step.calls if c.name not in _SENSORY]
            plans.append({"step": decision_index, "candidate_index": self.candidate_policies.index(selected_source),
                          "predicted_progress": value, "predicted_success_probability": pred_success,
                          "prediction_horizon_actions": self.prediction_horizon,
                          "predictions": predictions,
                          "executed_first_action": None if not actual_actions else _prefix_record(actual_actions[0])})
            if actual_step.error:
                stop_reason = actual_step.error
                break
            if not actual_actions:
                stop_reason = "candidate_completed"
                break
            candidate_action_counts[selected_index] += len(actual_actions)
            candidate_action_results[selected_index].extend(c.result for c in actual_actions)
            prefix.extend(_prefix_record(c) for c in actual_step.calls)
            decision_index += 1
            # Replanning always sees new current SDK state and the public call prefix.
        else:
            stop_reason = "actual_api_budget_exhausted"

        forced_failure = ("actual_api_budget_exhausted" if stop_reason == "actual_api_budget_exhausted"
                          else stop_reason if stop_reason.startswith("max_calls (") else None)
        final = _combine_final(ep, actual_traces, theta, seed, error=forced_failure)
        final.wall_time = time.perf_counter()-started
        total_calls = len(final.calls)
        last_candidate_index = plans[-1]["candidate_index"] if plans else None
        policy_source = (self.candidate_policies[last_candidate_index]
                         if last_candidate_index is not None else self.probe_source)
        return final, {"policy_source": policy_source,
                       "sim_predicted_success": None if last_predicted_success is None else last_predicted_success >= 0.5,
                       "runtime_sim_rollouts": runtime_sim_rollouts,
                       "runtime_replay_api_calls": runtime_replay_calls,
                       "runtime_sim_time_s": runtime_sim_time,
                       "runtime_sim_budget_per_episode": self.runtime_sim_budget_per_episode,
                       "prediction_horizon_actions": self.prediction_horizon,
                       "planning_steps": plans, "stop_reason": stop_reason,
                       "actual_api_calls": total_calls, "runtime_llm_calls": 0,
                       "runtime_tokens": 0, "wall_time_s": final.wall_time}


def run_twin_mpc_receding(backend: TaskBackend, llm: LLMClient, *, seed: int = 0,
                          workers: int = 1, max_rollouts: int = 300,
                          max_llm_calls: int = 8, n_twins: int = 16,
                          prediction_horizon: int = 3, top_k_twins: int = 2,
                          runtime_sim_budget_per_episode: int = 24) -> BaselineResult:
    """Build a particle/candidate bank, then replan by finite-horizon public-prefix replay."""
    if int(prediction_horizon) < 2:
        raise ValueError("Twin MPC prediction_horizon must be at least 2")
    trained = run_twin_mpc(backend, llm, seed=seed, workers=workers,
                           max_rollouts=max_rollouts, max_llm_calls=max_llm_calls,
                           n_twins=n_twins, runtime_sim_budget_per_episode=runtime_sim_budget_per_episode,
                           receding_horizon=True)
    old_agent = trained.agent
    if old_agent is None:
        raise RuntimeError("Twin MPC development did not produce a runtime agent")
    agent = TwinMPCRecedingAgent(old_agent.probe_source, old_agent.twin_bank,
                                 old_agent.probe_obs_bank, old_agent.candidate_policies,
                                 prediction_horizon=prediction_horizon, top_k_twins=top_k_twins,
                                 runtime_sim_budget_per_episode=runtime_sim_budget_per_episode)
    trained.method = "twin_mpc_receding"
    trained.agent = agent
    trained.extra.update({"identity": "custom_control", "implementation": "particle_twin_receding_horizon_mpc",
                          "prediction_horizon_actions": prediction_horizon,
                          "horizon_score": "weighted mean of task evaluate score after H forecast actions",
                          "first_action_only": True, "public_prefix_replay": True,
                          "replay_exposes_theta": False,
                          "candidate_continuation": "candidate-local actuator ordinal skipping with cached public action results; fresh SDK getters re-evaluate branches, but Python locals/stack are not resumed",
                          "paired_twin_seed": "seed + 50000 + particle_index, fixed across candidates and replanning steps",
                          "development_builder": "candidate/particle construction shared with legacy twin_mpc, with receding execution contract in repair prompts"})
    return trained
