"""Legacy online-probe proxy, registered as `physcap` for result compatibility.

This implementation does not reproduce PhysCaP's Planner/Prioritizer loop or
its physical measurement protocols. Use the proxy name in research comparisons.
"""

from __future__ import annotations

import math
import subprocess
import time
import threading
from typing import Any, Sequence

from rebranch.api.primitives import PHYSCAP_EXTRA_PRIMITIVES, RobotAPI
from rebranch.baselines import (
    BaselineResult,
    build_safe_probe_source,
    combine_two_phase_episode,
    finalize_episode_trace,
    merge_two_phase_traces,
)
from rebranch.exec.sandbox import StaticCheckError, check_policy_source, run_policy
from rebranch.exec.trace import Trace
from rebranch.eval.metrics import summarize_trace_for_llm
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.llm.prompts import PHYSCAP_EXTRA_DOC, SYSTEM_PROMPT, build_physcap_prompt
from rebranch.sim.backend import Episode, EpisodeResult, TaskBackend


class PhysCaPAPIWrapper(RobotAPI):
    """Wraps `RobotAPI` adding SDK-only `get_mass()` and `get_stiffness()` primitives."""

    def __init__(self, inner: RobotAPI) -> None:
        self._inner = inner
        self._last_ee_pos: list[float] | None = None
        self._force: float | None = None
        self._unloaded: list[tuple[list[float], list[float], float]] = []
        self._objects: dict[str, list[float]] = {}
        self._grasp_name: str | None = None
        self.measurements: list[dict[str, Any]] = []

    def get_ee_pose(self) -> dict[str, list[float]]:
        res = self._inner.get_ee_pose()
        self._last_ee_pos = list(res["pos"])
        return res

    def zero_ee_wrench(self) -> None:
        self._inner.zero_ee_wrench()
        self._unloaded.clear()  # A new tare defines a new measurement frame.

    def get_ee_wrench(self, frame: str = "world") -> dict[str, list[float]]:
        return self._inner.get_ee_wrench(frame=frame)

    def get_object_pose(self, name: str) -> dict[str, list[float]]:
        pose = self._inner.get_object_pose(name)
        if self._force is None:
            self._objects[name] = list(pose["pos"])
        return pose

    def open_gripper(self, width: float = 0.07) -> None:
        self._inner.open_gripper(width=width)
        self._force = None
        self._grasp_name = None

    def close_gripper(self, force: float = 8.0) -> None:
        if self._force is None and self._objects:
            ee = self._inner.get_ee_pose()["pos"]
            distances = {name: sum((self._inner.get_object_pose(name)["pos"][i] - ee[i])**2 for i in range(3)) for name in self._objects}
            nearest = min(distances, key=distances.get)
            self._grasp_name = nearest if distances[nearest] < .04**2 else None
        self._inner.close_gripper(force=force)
        self._force = float(force)

    def get_gripper_state(self) -> dict[str, Any]:
        return self._inner.get_gripper_state()

    def move_to(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.08,
    ) -> None:
        self._inner.move_to(pos, quat=quat, speed=speed)
        if self._force is None:
            samples = self._sample_hold()
            pose = self._inner.get_ee_pose()
            self._unloaded.append((list(pose["pos"]), list(pose["quat"]), samples["fz"]))
            self.measurements.append({"kind": "unloaded_reference", "sample_count": 12,
                                      "sample_interval_s": 0.02, "hold_s": 0.24,
                                      "pos": pose["pos"], "fz": samples["fz"]})

    def move_until_contact(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        speed: float = 0.04,
        max_force: float = 6.0,
        max_torque: float = 0.8,
    ) -> dict[str, Any]:
        return self._inner.move_until_contact(
            pos, quat=quat, speed=speed, max_force=max_force, max_torque=max_torque
        )

    def move_compliant(
        self,
        pos: Sequence[float],
        quat: Sequence[float] | None = None,
        free_axes: str = "",
        speed: float = 0.04,
        duration: float | None = None,
    ) -> None:
        self._inner.move_compliant(
            pos, quat=quat, free_axes=free_axes, speed=speed, duration=duration
        )

    def _sample_hold(self, n: int = 12) -> dict[str, float]:
        """Average distinct SDK samples, spaced by explicit hold actions, while holding pose."""
        pose = self._inner.get_ee_pose()
        rows = []
        for _ in range(n):
            pose = self._inner.get_ee_pose()
            self._inner.move_compliant(pose["pos"], quat=pose["quat"], duration=0.02)
            gs = self._inner.get_gripper_state()
            rows.append((gs["width"], gs["effort"], self._inner.get_ee_wrench()["force"][2]))
        return dict(zip(("width", "effort", "fz"), (sum(r[i] for r in rows) / n for i in range(3))))

    def get_mass(self) -> float | None:
        """Paired unloaded/loaded measurement at the same pose; None if unsupported."""
        pos = self._inner.get_ee_pose()["pos"]
        quat = self._inner.get_ee_pose()["quat"]
        refs = [(p, q, fz) for p, q, fz in self._unloaded
                if sum((p[i] - pos[i]) ** 2 for i in range(3)) < 0.01 ** 2
                and abs(sum(q[i] * quat[i] for i in range(4))) >= math.cos(math.radians(5.0))]
        follows = (self._grasp_name is not None and
                   self._inner.get_object_pose(self._grasp_name)["pos"][2]
                   - self._objects[self._grasp_name][2] >= 0.02)
        if self._force is None or not refs or not follows:
            self.measurements.append({"kind": "mass", "value": None, "reason": "needs_paired_pose_and_confirmed_lift"})
            return None
        reference, ref_quat, baseline = min(refs, key=lambda r: sum((r[0][i]-pos[i])**2 for i in range(3)))
        # Correct load-induced tracking offset using SDK pose feedback, rather
        # than weakening the 2 mm pairing requirement.
        corrections = []
        for _ in range(2):
            actual = self._inner.get_ee_pose()["pos"]
            error = [reference[i] - actual[i] for i in range(3)]
            actual_quat = self._inner.get_ee_pose()["quat"]
            alignment = abs(sum(actual_quat[i] * ref_quat[i] for i in range(4)))
            if sum(e*e for e in error) <= 0.001 ** 2 and alignment >= math.cos(math.radians(.5)):
                break
            def product(a, b):
                w,x,y,z=a; v,i,j,k=b
                return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]
            inverse_actual = [actual_quat[0], *[-x for x in actual_quat[1:]]]
            target_quat = product(product(ref_quat, inverse_actual), ref_quat)
            target = [reference[i] + error[i] for i in range(3)]
            self._inner.move_to(target, quat=target_quat, speed=0.01)
            corrections.append({"before": actual, "target": target, "target_quat": target_quat})
        actual = self._inner.get_ee_pose()["pos"]
        pair_error = sum((reference[i] - actual[i]) ** 2 for i in range(3)) ** .5
        paired_quat = self._inner.get_ee_pose()["quat"]
        angle_error = 2 * math.acos(min(1., abs(sum(paired_quat[i] * ref_quat[i] for i in range(4)))))
        if pair_error > .002 or angle_error > math.radians(1.):
            self.measurements.append({"kind": "mass", "value": None, "reason": "pose_pair_out_of_tolerance", "pair_error_m": pair_error})
            return None
        before = {self._grasp_name: self._inner.get_object_pose(self._grasp_name)["pos"]}
        loaded = self._sample_hold()["fz"]
        stable = any(
            self._inner.get_object_pose(name)["pos"][2] - self._objects[name][2] >= 0.02
            and abs(self._inner.get_object_pose(name)["pos"][2] - p[2]) < 0.002
            for name, p in before.items()
        )
        if not stable:
            self.measurements.append({"kind": "mass", "value": None, "reason": "payload_moved_during_sampling"})
            return None
        mass = (baseline - loaded) / 9.81
        self.measurements.append({"kind": "mass", "value": mass, "loaded_fz": loaded, "unloaded_fz": baseline, "pos": actual, "pair_error_m": pair_error, "pair_angle_error_rad": angle_error, "corrections": corrections})
        return mass

    def get_stiffness(self) -> float | None:
        """Secant squeeze stiffness from two settled force/width measurements.

        Width noise/quantization sets a resolution floor. An unresolved rigid
        response is unavailable, not zero stiffness or an arbitrary clipped number.
        """
        force = self._force
        name = self._grasp_name
        supported = (name is not None and
            abs(self._inner.get_object_pose(name)["pos"][2] - self._objects[name][2]) <= 0.002)

        if force is None or not supported:
            self.measurements.append({"kind": "stiffness", "value": None,
                                      "reason": "requires_object_at_observed_support_height"})
            return None
        self._inner.close_gripper(force=1.0)
        a = self._sample_hold()
        self._inner.close_gripper(force=3.0)
        b = self._sample_hold()
        self._inner.close_gripper(force=force)
        dw = a["width"] - b["width"]
        df = b["effort"] - a["effort"]
        value = df / dw if dw > 0.0005 and df > 0.5 else None
        self.measurements.append({"kind": "stiffness", "value": value, "low": a, "high": b, "delta_width": dw, "resolution_m": 0.0005})
        return value


class PhysCaPEpisodeWrapper(Episode):
    """Episode wrapper exposing `PhysCaPAPIWrapper` while preserving `snapshot()`/`evaluate()`."""

    def __init__(self, inner: Episode) -> None:
        self._inner = inner
        self.api = PhysCaPAPIWrapper(inner.api)
        self.protocol_version = getattr(inner, "protocol_version", "unversioned")

    @property
    def sim_time(self) -> float:
        return float(getattr(self._inner, "sim_time", 0.0))

    def snapshot(self) -> dict[str, float]:
        return self._inner.snapshot()

    def evaluate(self) -> EpisodeResult:
        return self._inner.evaluate()


def _augment_probe_with_physcap(probe_src: str) -> str:
    """Append `get_mass()` and `get_stiffness()` reads to a safe probe script."""
    lines = probe_src.rstrip().splitlines()
    # Read while grasped, before the probe's explicit release.
    release = lines.index("    # PROBE_FEATURES_V2")
    lines[release:release] = ["    api.get_stiffness()", "    api.get_mass()"]
    cand = "\n".join(lines) + "\n"
    check_policy_source(cand, extra_primitives=PHYSCAP_EXTRA_PRIMITIVES)
    return cand


_QUOTA_AUTH_TOKENS = (
    "quota",
    "rate limit",
    "ratelimit",
    "insufficient",
    "unauthorized",
    "401",
    "403",
    "auth",
    "login",
    "credit",
)


def _is_fatal_llm_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in _QUOTA_AUTH_TOKENS)


class PhysCaPAgent:
    """Online-probe proxy executing an initial probe + one online code generation."""

    def __init__(
        self,
        llm: LLMClient,
        base_policy: str,
        probe_policy: str,
        runtime_llm_budget: int = 100,
        max_requests_per_episode: int = 2,
    ) -> None:
        self.llm = llm
        self.base_policy = base_policy
        self.probe_policy = probe_policy
        self.runtime_llm_budget = max(0, int(runtime_llm_budget))
        self.max_requests_per_episode = max(0, int(max_requests_per_episode))
        self.runtime_llm_attempts = 0
        self._budget_lock = threading.Lock()

    def evaluate_episode(
        self,
        backend: TaskBackend,
        theta: dict[str, Any],
        seed: int = 0,
    ) -> tuple[Trace, dict[str, Any]]:
        t0 = time.perf_counter()
        ep = PhysCaPEpisodeWrapper(backend.make_episode(theta, seed=seed))

        # Phase 1: run gentle sensory probe inside the episode
        t_probe = run_policy(
            self.probe_policy,
            ep,
            theta,
            seed=seed,
            max_calls=200,
            extra_primitives=PHYSCAP_EXTRA_PRIMITIVES,
            evaluate=False,
        )
        if t_probe.error is not None:
            trace = finalize_episode_trace(ep, t_probe)
            return trace, {
                "policy_source": self.probe_policy,
                "sim_predicted_success": None,
                "runtime_llm_calls": 0,
                "runtime_tokens": 0,
                "wall_time_s": time.perf_counter() - t0,
            }

        # Summarize SDK observables + get_mass / get_stiffness (zero theta leak)
        mass_est = None
        stiff_est = None
        for c in t_probe.calls:
            if c.name == "get_mass" and isinstance(c.result, (int, float)):
                mass_est = float(c.result)
            elif c.name == "get_stiffness" and isinstance(c.result, (int, float)):
                stiff_est = float(c.result)

        obs_summary = (
            f"Online sensory probe readings:\n"
            f"  estimated_mass_kg={mass_est!r}\n"
            f"  estimated_stiffness_npm={stiff_est!r}\n"
            "  None means unmeasured/unresolved, not zero.\n"
            f"{summarize_trace_for_llm(t_probe, max_calls=8)}\n"
            f"SDK measurement protocol records: {ep.api.measurements!r}"
        )

        remaining_calls = 200 - len(t_probe.calls)
        if remaining_calls <= 0:
            trace = finalize_episode_trace(
                ep, t_probe,
                execution_error="max_calls (200) exhausted before action phase",
            )
            return trace, {
                "policy_source": self.base_policy,
                "sim_predicted_success": None,
                "runtime_llm_calls": 0,
                "runtime_tokens": 0,
                "fallback_reason": "episode_call_budget_exhausted_before_action",
                "wall_time_s": time.perf_counter() - t0,
                "measurements": ep.api.measurements,
            }

        # Phase 2: online LLM planning using SDK readings
        prompt = build_physcap_prompt(
            task_description=backend.task_description,
            current_code=self.base_policy,
            obs_summary=obs_summary,
            extra_primitives_doc=PHYSCAP_EXTRA_DOC,
        )
        resp = None
        runtime_llm_calls = 0
        runtime_tokens = 0
        request_events = []
        fallback_reason = None
        failed_requests = 0
        new_requests = 0
        for attempt in range(self.max_requests_per_episode):
            with self._budget_lock:
                if self.runtime_llm_attempts >= self.runtime_llm_budget:
                    fallback_reason = "runtime_llm_budget_exhausted"
                    break
                self.runtime_llm_attempts += 1
            runtime_llm_calls += 1
            try:
                resp = self.llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.2)
                runtime_tokens += int(resp.prompt_tokens) + int(resp.completion_tokens)
                new_requests += int(not resp.cached)
                request_events.append({"attempt": attempt + 1, "status": "completed", "cached": resp.cached,
                                       "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens})
                break
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                failed_requests += 1
                new_requests += 1  # Failed dispatch counted conservatively; billing may be unknown.
                fallback_reason = type(exc).__name__
                request_events.append({"attempt": attempt + 1, "status": "failed", "error": str(exc), "tokens": None})
                if isinstance(exc, RuntimeError) and _is_fatal_llm_error(exc):
                    break
        if resp is None and fallback_reason is None:
            fallback_reason = "runtime_llm_disabled"

        if resp is None:
            plan_src = self.base_policy
        else:
            plan_src = extract_python_code(resp.text)
            try:
                check_policy_source(plan_src, extra_primitives=PHYSCAP_EXTRA_PRIMITIVES)
            except StaticCheckError as exc:
                fallback_reason = f"invalid_generated_policy: {exc}"
                plan_src = self.base_policy

        t_act = run_policy(
            plan_src,
            ep,
            theta,
            seed=seed + 10000,
            max_calls=remaining_calls,
            extra_primitives=PHYSCAP_EXTRA_PRIMITIVES,
            evaluate=False,
        )
        trace = merge_two_phase_traces(ep, t_probe, t_act, max_calls=200)
        trace.wall_time = time.perf_counter() - t0
        return trace, {
            "policy_source": plan_src,
            "sim_predicted_success": None,
            "runtime_llm_calls": runtime_llm_calls,
            "runtime_tokens": None if failed_requests else runtime_tokens,
            "runtime_tokens_known": runtime_tokens,
            "runtime_llm_failed_attempts": failed_requests,
            "runtime_new_requests": new_requests,
            "runtime_request_events": request_events,
            "runtime_llm_budget": self.runtime_llm_budget,
            "runtime_llm_budget_used": self.runtime_llm_attempts,
            "runtime_llm_budget_scope": "agent lifetime across test episodes; separate from development",
            "fallback_reason": fallback_reason,
            "wall_time_s": trace.wall_time,
            "measurements": ep.api.measurements,
        }


def run_physcap(
    backend: TaskBackend,
    llm: LLMClient,
    *,
    max_dev_llm_calls: int = 1,
    runtime_llm_budget: int = 100,
    max_requests_per_episode: int = 2,
) -> BaselineResult:
    """Initialize PhysCaP online agent (optional 1 dev LLM call to adapt draft)."""
    t0 = time.perf_counter()
    probe_src = _augment_probe_with_physcap(build_safe_probe_source(backend))
    base_src = backend.draft_policy
    dev_calls = 0
    dev_pt = 0
    dev_ct = 0

    if max_dev_llm_calls > 0:
        prompt = build_physcap_prompt(
            task_description=backend.task_description,
            current_code=backend.draft_policy,
            obs_summary="Offline initialization before online execution.",
            extra_primitives_doc=PHYSCAP_EXTRA_DOC,
        )
        resp = llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.2)
        dev_calls = 1
        dev_pt = int(resp.prompt_tokens)
        dev_ct = int(resp.completion_tokens)
        cand = extract_python_code(resp.text)
        try:
            check_policy_source(cand, extra_primitives=PHYSCAP_EXTRA_PRIMITIVES)
            base_src = cand
        except StaticCheckError:
            base_src = backend.draft_policy

    agent = PhysCaPAgent(llm=llm, base_policy=base_src, probe_policy=probe_src, runtime_llm_budget=runtime_llm_budget, max_requests_per_episode=max_requests_per_episode)
    return BaselineResult(
        method="physcap",
        policy_source=base_src,
        n_rollouts=0,
        dev_llm_calls=dev_calls,
        dev_prompt_tokens=dev_pt,
        dev_completion_tokens=dev_ct,
        agent=agent,
        program_snapshots=[{"round": 0, "accepted": True, "source": base_src}],
        wall_time_s=time.perf_counter() - t0,
        extra={"implementation": "online_probe_proxy", "reference_fidelity": "not_physcap_reproduction", "dev_llm_budget": max_dev_llm_calls, "runtime_llm_budget": runtime_llm_budget, "max_requests_per_episode": max_requests_per_episode},
    )
