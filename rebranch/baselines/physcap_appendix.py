"""Appendix-oriented PhysCaP simulation baseline; separate from legacy proxy."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from rebranch.api.primitives import API_DOC
from rebranch.baselines import BaselineResult
from rebranch.exec.sandbox import StaticCheckError, run_policy
from rebranch.exec.trace import Trace
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.sim.backend import TaskBackend

ASSETS = Path(__file__).with_name("physcap_appendix_assets") / "prompts"
MAX_API_CALLS = 200
MAX_POLICY_LINES = 50_000


class _MassAPI:
    """Expose the gated simulator getter as one additional policy primitive."""
    def __init__(self, inner: Any, episode: Any, home_pose: dict[str, Any]):
        self._inner = inner
        self._episode = episode
        self._home_pose = home_pose
        self.measurements: list[dict[str, Any]] = []

    def get_mass(self, name: str) -> float:
        """Official high-level grasp/lift/hold/read/release mass macro."""
        started = float(getattr(self._episode, "sim_time", 0.0))
        object_pose = self._inner.get_object_pose(name)
        pos = list(object_pose["pos"])
        quat = list(self._home_pose["quat"])
        approach = list(pos)
        approach[2] += 0.10
        above = list(pos)
        above[2] += 0.15
        self._inner.open_gripper()
        self._inner.move_to(approach, quat=quat)
        self._inner.move_to(pos, quat=quat)
        self._inner.close_gripper()
        self._inner.move_to(above, quat=quat)
        self._inner.move_compliant(above, quat=quat, duration=0.6)
        raw_value = self._episode.get_mass(name)
        value = float(raw_value) if raw_value is not None else float("nan")
        place = list(pos)
        place[2] += 0.005
        self._inner.move_to(place, quat=quat)
        self._inner.move_compliant(place, quat=quat, duration=0.3)
        self._inner.open_gripper()
        self._inner.move_to(above, quat=quat)
        self._inner.move_to(self._home_pose["pos"], quat=self._home_pose["quat"])
        valid = math.isfinite(value)
        record = {"name": str(name), "value": value if valid else None,
                  "sim_time_s": float(getattr(self._episode, "sim_time", 0.0)) - started,
                  "revealed": valid, "return_status": "measured" if valid else "gate_not_passed"}
        self.measurements.append(record)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _MassEpisode:
    def __getattr__(self, name: str):
        """Preserve inner Episode safety, termination and task-diagnostic access."""
        return getattr(self._inner, name)

    def __init__(self, inner: Any):
        self._inner = inner
        self.initial_ee_pose = inner.api.get_ee_pose()
        self.api = _MassAPI(inner.api, inner, self.initial_ee_pose)
        self.extra_primitives = ("get_mass",)
        self.protocol_version = getattr(inner, "protocol_version", "unversioned")

    @property
    def sim_time(self) -> float:
        return float(getattr(self._inner, "sim_time", 0.0))

    def snapshot(self):
        return self._inner.snapshot()

    def evaluate(self):
        return self._inner.evaluate()

    def render_rgb(self, **kwargs):
        return self._inner.render_rgb(**kwargs)


def with_physcap_mass_macro(inner: Any) -> _MassEpisode:
    """Attach the shared Appendix A.4 gate and 15 cm measurement macro to an episode."""
    inner.enable_physcap_mass_reveal()
    return _MassEpisode(inner)


class PhysCaPTaskBackend(TaskBackend):
    """Backend adapter enabling appendix mass gate on each MuJoCo episode."""
    def __init__(self, inner: TaskBackend):
        self._inner = inner
        for name in ("name", "task_description", "requires_branching", "param_space",
                     "obs_scale", "obs_api_map", "draft_policy"):
            setattr(self, name, getattr(inner, name))

    def make_episode(self, theta, seed=0):
        episode = self._inner.make_episode(theta, seed=seed)
        if "get_mass" in getattr(episode, "extra_primitives", ()):
            return episode
        return with_physcap_mass_macro(episode)

    def oracle_policy(self, theta):
        return self._inner.oracle_policy(theta)

    def hidden_test_set(self, seed=2026, n=50):
        return self._inner.hidden_test_set(seed=seed, n=n)

    def ood_test_set(self, seed=3026, n=10):
        return self._inner.ood_test_set(seed=seed, n=n)

    def conservative_policy(self):
        return self._inner.conservative_policy()

    def latent_mode(self, theta):
        return self._inner.latent_mode(theta)


def _asset(path: str) -> str:
    return (ASSETS / path).read_text(encoding="utf-8")


def _json_value(text: str, expected: type) -> Any:
    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if not match:
        raise ValueError("model response contained no JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, expected):
        raise ValueError(f"model JSON must be {expected.__name__}")
    return value


def _normalize_candidate_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("_", " ").replace("-", " ").strip().lower()


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    name = _normalize_candidate_name(candidate.get("name"))
    action = _normalize_candidate_name(candidate.get("action"))
    if name == "original task execute":
        return (name, "")
    if not action or action == "other":
        action = _normalize_candidate_name(candidate.get("description"))
    return (name, action)


def _rank_scored_candidates(original: list[dict[str, Any]], ranked_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the official score-mode schema and stably rank original rows."""
    rows = ranked_data.get("scored_candidates")
    if not isinstance(rows, list) or len(rows) != len(original):
        raise ValueError("prioritizer scored_candidates must score every pending item")
    original_by_key = {_candidate_key(candidate): candidate for candidate in original}
    scores: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prioritizer score rows must be objects")
        key = _candidate_key(row)
        score = row.get("priority_score")
        if (key not in original_by_key or key in scores
                or not isinstance(row.get("reason"), str)
                or isinstance(score, bool) or not isinstance(score, int)
                or not 0 <= score <= 10):
            raise ValueError("prioritizer returned invalid score-mode fields or candidate identity")
        scores[key] = score
    if set(scores) != set(original_by_key):
        raise ValueError("prioritizer scores do not match original candidate set")
    return sorted(original, key=lambda candidate: -scores[_candidate_key(candidate)])


def _stable_rank(original: list[dict[str, Any]], ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [_candidate_key(c) for c in original]
    requested = [_candidate_key(c) for c in ranked]
    if len(set(keys)) != len(keys) or len(requested) != len(keys) or set(requested) != set(keys):
        raise ValueError("prioritizer response must be an exact permutation of candidate (name, action) pairs")
    by_key = dict(zip(keys, original))
    return [by_key[key] for key in requested]


def _format(template: str, *, task: str, knowledge: str, history: str, candidate: dict[str, Any] | None = None) -> str:
    candidate = candidate or {}
    values = {
        "{candidate_name}": str(candidate.get("name", "original task")),
        "{candidate_action}": str(candidate.get("action", "complete_task")),
        "{candidate_desc}": str(candidate.get("description", task)),
        "{candidate_info}": str(candidate.get("expected_info", "")),
        "{candidate_cost}": str(candidate.get("estimated_cost", "")),
        "{candidate_params}": json.dumps(candidate.get("parameters", {})),
        "{original_task}": task,
        "{scene_knowledge}": knowledge,
        "{interaction_history}": history,
        "{global_knowledge}": knowledge,
    }
    for token, value in values.items():
        template = template.replace(token, value)
    return template


class PhysCaPAppendixAgent:
    """Per-episode visual Planner, independent Prioritizer, and one-step coder."""

    def __init__(self, llm: LLMClient, *, max_turns: int = 10, max_calls: int = MAX_API_CALLS):
        self.llm = llm
        self.max_turns = int(max_turns)
        self.max_calls = int(max_calls)

    def evaluate_episode(self, backend: TaskBackend, theta: dict[str, Any], seed: int = 0) -> tuple[Trace, dict[str, Any]]:
        started = time.perf_counter()
        episode = backend.make_episode(theta, seed=seed)
        task = str(backend.task_description)
        events: list[dict[str, Any]] = []
        phases: list[Trace] = []
        history: list[str] = []
        knowledge = ""
        latest_visual_difference = ""
        selected_sources: list[str] = []
        pool: dict[tuple[str, str], dict[str, Any]] = {}
        executed: set[tuple[str, str]] = set()
        measurements: list[dict[str, Any]] = []
        unrecorded_attempts: dict[tuple[str, str], int] = {}
        failed_candidates: list[tuple[str, str]] = []
        prioritizer_fallbacks: list[dict[str, Any]] = []
        runtime_error: str | None = None
        stop_reason = "turn_limit"

        def complete(role: str, prompt: str, images: tuple[bytes, ...] = ()) -> str:
            event = {"role": role, "prompt": prompt, "images_sha256": [hashlib.sha256(i).hexdigest() for i in images],
                     "prompt_tokens": None, "completion_tokens": None, "cached": False,
                     "status": "attempted", "response": None, "error": None}
            events.append(event)
            try:
                response = self.llm.complete(prompt, images=images)
            except Exception as exc:
                event["status"] = "error"
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"{role} request failed: {event['error']}") from exc
            event.update(status="completed", response=response.text,
                         prompt_tokens=response.prompt_tokens,
                         completion_tokens=response.completion_tokens, cached=response.cached)
            return response.text

        def phase(source: str) -> Trace:
            remaining = self.max_calls - sum(len(t.calls) for t in phases)
            try:
                return run_policy(source, episode, theta, seed=seed, max_calls=remaining,
                                  max_lines=MAX_POLICY_LINES, extra_primitives=("get_mass",),
                                  evaluate=False, capture_output=True)
            except StaticCheckError as exc:
                return Trace(
                    theta=dict(theta), seed=seed, success=False, score=-1.0,
                    failure=str(exc), error=f"StaticCheckError: {exc}",
                    sim_time=float(getattr(episode, "sim_time", 0.0)),
                    protocol_version=getattr(episode, "protocol_version", "unversioned"),
                )

        def coding_api_doc() -> str:
            return (API_DOC + "\n\nPhysCaP simulation high-level measurement API:\n"
            "Define exactly `def run(api):` as the entry point.\n"
            "For API object arguments, use the exact public object IDs explicitly named in the task description. "
            "Do not inspect or infer hidden labels or theta.\n"
            "api.get_mass(name) performs the official interaction macro: approach, grasp, "
            "lift 0.15 m, hold 0.6 s, read simulator mass only after the per-step 0.5 s "
            "grip and 0.5 s lift gates pass, then lower to 5 mm above the original pose, "
            "settle, release, retreat, and return to the episode-initial home pose. It "
            "returns NaN when the gate did not pass; NaN is an unsuccessful measurement. Print any "
            "measurement values and print FINISH only when the task is complete. Each "
            "RobotAPI call, including get_mass, counts toward the shared 200-call episode cap.")

        def execute_full_task(plan: dict[str, Any]) -> None:
            prompt = (coding_api_doc() + "\n\n" + _format(
                _asset("coding_agent/full_task_with_knowledge.txt"), task=task,
                knowledge=knowledge, history="\n".join(history))
                + "\n\nVisual-differencing results:\n" + latest_visual_difference
                + "\n\nPrevious generated code and stdout/stderr:\n" + "\n".join(history))
            code = extract_python_code(complete("coding_full_task", prompt))
            if not code:
                raise ValueError("coding agent returned no policy")
            selected_sources.append(code)
            phases.append(phase(code))
            if phases[-1].error:
                raise RuntimeError(f"coding phase error: {phases[-1].error}")
            stop_reason_local = "finish_signal" if "FINISH" in phases[-1].stdout else "full_task_policy"
            history.append(f"full_task code={code}\nstdout={phases[-1].stdout}\nstderr={phases[-1].stderr}")
            nonlocal_stop[0] = stop_reason_local

        nonlocal_stop = [""]
        try:
            frame = episode.render_rgb()
            knowledge = complete("scene_knowledge", _asset("scene_knowledge.txt") + "\n\nTask:\n" + task, (frame,))
            for _turn in range(self.max_turns):
                frame = episode.render_rgb()
                planner_prompt = (_asset("planner.txt") + "\n\nTask:\n" + task
                                  + "\n\nUse the exact public object IDs explicitly named in the task description for API calls. Do not inspect or infer hidden labels or theta."
                                  + "\n\nAccumulated knowledge:\n" + knowledge
                                  + "\n\nVerified gated mass results (authoritative; null means measurement not obtained):\n"
                                  + json.dumps(measurements, allow_nan=False)
                                  + "\n\nInteraction history:\n" + "\n".join(history))
                plan = _json_value(complete("planner", planner_prompt, (frame,)), dict)
                if plan.get("task_complete") and phases:
                    stop_reason = "task_complete"
                    break
                if plan.get("sufficient"):
                    execute_full_task(plan)
                    stop_reason = nonlocal_stop[0]
                    break

                candidates = plan.get("exploration_candidates", [])
                if not isinstance(candidates, list):
                    raise ValueError("Planner candidates must be a list")
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        pool.setdefault(_candidate_key(candidate), candidate)
                pending = [c for key, c in pool.items() if key not in executed]
                if not pending:
                    execute_full_task(plan)
                    stop_reason = nonlocal_stop[0]
                    break
                schema = {
                    "type": "object", "required": ["scored_candidates"],
                    "properties": {"scored_candidates": {"type": "array", "items": {
                        "type": "object", "required": ["name", "action", "reason", "priority_score"],
                        "properties": {"name": {"type": "string"}, "action": {"type": "string"},
                                       "reason": {"type": "string"}, "priority_score": {
                                           "type": "integer", "minimum": 0, "maximum": 10}},
                    }}},
                }
                ranking_note = ("\n\nscore mode: score each candidate independently from 0 to 10 "
                                "(10=highest priority, 0=lowest). Return ONLY a JSON object matching this schema; "
                                "preserve each candidate's name and action, give a short reason, and include every candidate.\n"
                                + json.dumps(schema, indent=2))
                rank_prompt = (_asset("prioritizer.txt") + ranking_note + "\n\nTask description and public object IDs:\n"
                               + task + "\n\nCandidates:\n"
                               + json.dumps(pending) + "\n\nScene knowledge:\n" + knowledge)
                ranked = None
                invalid_attempts = []
                for attempt in range(1, 4):
                    try:
                        ranked_data = _json_value(complete("prioritizer", rank_prompt, (frame,)), dict)
                        ranked = _rank_scored_candidates(pending, ranked_data)
                    except (ValueError, json.JSONDecodeError) as exc:
                        invalid_attempts.append(f"attempt {attempt}: {exc}")
                        events[-1]["validation_error"] = str(exc)
                        continue
                    break
                if ranked is None:
                    ranked = list(pending)
                    prioritizer_fallbacks.append({"status": "fallback_original_order",
                                                  "attempts": 3, "invalid_attempts": invalid_attempts})
                selected = ranked[0]
                code_prompt = (coding_api_doc() + "\n\n" + _format(_asset("coding_agent/single_step.txt"),
                               task=task, knowledge=knowledge, history="\n".join(history), candidate=selected)
                               + "\n\nPrior code and stdout/stderr:\n" + (history[-1] if history else "")
                               + "\n\nVisual-differencing results:\n" + latest_visual_difference)
                code = extract_python_code(complete("coding_single_step", code_prompt))
                if not code:
                    raise ValueError("coding agent returned no policy")
                selected_sources.append(code)
                this_phase = phase(code)
                phases.append(this_phase)
                selected_key = _candidate_key(selected)
                phase_measurements = list(episode.api.measurements)
                measurements.extend(phase_measurements)
                episode.api.measurements.clear()
                measured_names = {_normalize_candidate_name(row.get("name")) for row in phase_measurements
                                  if row.get("revealed") and row.get("value") is not None}
                measurement_recorded = _candidate_key(selected)[0] in measured_names
                # Every actual mass read retires its mass candidate, including
                # reads bundled with another action or followed by a later error.
                for measured_name in measured_names:
                    executed.add((measured_name, "measure mass"))
                terminated_call = next((call for call in this_phase.calls
                                        if isinstance(call.result, dict) and "terminated" in call.result), None)
                if terminated_call is not None:
                    stop_reason = "episode_terminated"
                    runtime_error = str(terminated_call.result["terminated"])
                    break
                if this_phase.error and "max_calls" in str(this_phase.error):
                    stop_reason = "api_call_budget_exhausted"
                    runtime_error = "shared RobotAPI call budget exhausted"
                    break
                if this_phase.error and "max_lines" in str(this_phase.error):
                    stop_reason = "program_line_budget_exhausted"
                    runtime_error = this_phase.error
                    break
                history.append(f"candidate={selected!r}\ncode={code}\nstdout={this_phase.stdout}\nstderr={this_phase.stderr}\nerror={this_phase.error}")
                if "FINISH" in this_phase.stdout:
                    stop_reason = "finish_signal"
                    break
                if sum(len(t.calls) for t in phases) >= self.max_calls:
                    stop_reason = "api_call_budget_reached"
                    break
                if this_phase.error:
                    if selected_key not in failed_candidates:
                        failed_candidates.append(selected_key)
                    unrecorded_attempts[selected_key] = unrecorded_attempts.get(selected_key, 0) + 1
                    if unrecorded_attempts[selected_key] >= 2:
                        executed.add(selected_key)
                elif _candidate_key(selected)[1] == "measure mass" and not measurement_recorded:
                    unrecorded_attempts[selected_key] = unrecorded_attempts.get(selected_key, 0) + 1
                    if unrecorded_attempts[selected_key] >= 2:
                        executed.add(selected_key)
                elif _candidate_key(selected)[1] != "measure mass":
                    executed.add(selected_key)
                after = episode.render_rgb()
                knowledge += ("\n\n[Step Update]\nAction: " + json.dumps(selected)
                              + "\nSDK stdout:\n" + this_phase.stdout
                              + "\nSDK stderr:\n" + this_phase.stderr
                              + "\nPhysCaP gated simulator measurements (null means not obtained):\n"
                              + json.dumps(phase_measurements, allow_nan=False))
                vdm_prompt = (
                    "Describe only physical changes between BEFORE and AFTER images. "
                    "Be concise and factual; state what new information was revealed.\n\n"
                    f"Task context: {task}\nAction taken: {json.dumps(selected)}"
                )
                latest_visual_difference = complete("visual_differencing", vdm_prompt, (frame, after))
                knowledge += "\nVisual difference: " + latest_visual_difference
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            if " request failed:" in str(exc):
                stop_reason = str(exc).split(" request failed:", 1)[0] + "_request_error"
            elif stop_reason == "turn_limit":
                stop_reason = "agent_error"

        # Every runtime path performs one physical evaluation after all phases.
        result = episode.evaluate()
        call_count = sum(len(t.calls) for t in phases)
        final_policy_error = next((t.error for t in phases[-1:] if t.error), None) if stop_reason in {"full_task_policy", "finish_signal"} and phases else None
        error = runtime_error or final_policy_error
        if call_count > self.max_calls:
            error = error or f"max_calls ({self.max_calls}) exceeded"
            stop_reason = "api_call_budget_exhausted"
        success = bool(result.success and error is None)
        score = result.score if success else (min(result.score, -0.5) if error else result.score)
        calls = []
        failure_call = None
        for trace in phases:
            offset = len(calls)
            calls.extend(replace(
                call, index=offset + i,
                result=(None if isinstance(call.result, (int, float)) and not math.isfinite(float(call.result)) else call.result),
            ) for i, call in enumerate(trace.calls))
            if failure_call is None and trace.failure_call is not None:
                failure_call = offset + trace.failure_call
        measurements.extend(episode.api.measurements)
        trace = Trace(theta=dict(theta), seed=seed, calls=calls,
                      lines=[line for t in phases for line in t.lines],
                      success=success, score=score,
                      failure=error or result.failure, failure_call=failure_call,
                      safety_violation=result.safety_violation, error=error,
                      sim_time=float(getattr(episode, "sim_time", 0.0)),
                      wall_time=time.perf_counter() - started,
                      protocol_version=getattr(episode, "protocol_version", "unversioned"),
                      stdout="\n".join(t.stdout for t in phases),
                      stderr="\n".join(t.stderr for t in phases))
        source = "\n\n".join(selected_sources)
        metadata = {"policy_source": source, "sim_predicted_success": None,
                    "runtime_llm_calls": len(events),
                    "runtime_tokens": (sum(e["prompt_tokens"] + e["completion_tokens"] for e in events)
                                       if all(e["prompt_tokens"] is not None for e in events) else None),
                    "runtime_sim_rollouts": 0, "runtime_sim_time_s": trace.sim_time,
                    "wall_time_s": trace.wall_time, "stop_reason": stop_reason,
                    "episode_initial_home_pose": getattr(episode, "initial_ee_pose", None),
                    "measurements": measurements, "llm_events": events,
                    "failed_candidates": failed_candidates,
                    "prioritizer_fallbacks": prioritizer_fallbacks,
                    "unrecorded_attempts": {str(k): v for k, v in unrecorded_attempts.items()}}
        return trace, metadata


def run_physcap_appendix(backend: TaskBackend, llm: LLMClient, *, max_turns: int = 10) -> BaselineResult:
    """Build the runtime agent; development performs no LLM calls or rollouts."""
    return BaselineResult(method="physcap_appendix", policy_source="", n_rollouts=0,
                          dev_llm_calls=0, dev_prompt_tokens=0, dev_completion_tokens=0,
                          agent=PhysCaPAppendixAgent(llm, max_turns=max_turns),
                          extra={"implementation_identity": "physcap-appendix-simulation-mass-v1",
                                 "dev_rollout_budget": 0})
