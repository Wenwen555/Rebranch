"""AST static safety checker and traced sandbox runner for policies."""

from __future__ import annotations

import ast
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
import math
import sys
import time
from typing import Any, Sequence

from rebranch.api.primitives import PRIMITIVES, EpisodeTerminated
from rebranch.exec.trace import CallRecord, Trace
from rebranch.sim.backend import Episode


class StaticCheckError(ValueError):
    """Raised when policy source violates AST sandbox rules."""


class PlanningHorizonReached(Exception):
    """Internal stop after a forecast horizon; not a policy execution failure."""


FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "__import__",
        "compile",
        "input",
        "breakpoint",
        "memoryview",
        "type",
        "object",
        "__builtins__",
    }
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "False": False,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "None": None,
    "pow": pow,
    "range": range,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "True": True,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}


def check_policy_source(
    source: str, extra_primitives: Sequence[str] = ()
) -> None:
    """Validate policy Python source against AST sandbox rules."""
    _ = extra_primitives
    try:
        tree = ast.parse(source, filename="<policy>")
    except SyntaxError as exc:
        raise StaticCheckError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc

    has_run = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            n_pos = len(node.args.posonlyargs) + len(node.args.args)
            if n_pos < 1 and node.args.vararg is None:
                raise StaticCheckError("Function `run` must accept `api` argument")
            has_run = True

    if not has_run:
        raise StaticCheckError("Policy must define top-level `def run(api):`")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise StaticCheckError(
                f"Import statements are forbidden (line {getattr(node, 'lineno', '?')})"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise StaticCheckError(
                f"Private attribute access {node.attr!r} is forbidden (line {node.lineno})"
            )
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise StaticCheckError(
                    f"Forbidden identifier {node.id!r} (line {node.lineno})"
                )


def _policy_caller_lineno(max_depth: int = 16) -> int:
    """Walk stack frames to find the caller line number inside `<policy>`."""
    frame = sys._getframe(1)
    for _ in range(max_depth):
        if frame is None:
            break
        if frame.f_code.co_filename == "<policy>":
            return int(frame.f_lineno)
        frame = frame.f_back
    return 0


class _TracedAPI:
    """Proxy wrapping `RobotAPI` that enforces allowed primitives and logs calls."""

    def __init__(
        self,
        episode: Episode,
        allowed: frozenset[str],
        max_calls: int,
        planning_horizon_actions: int | None = None,
        skip_actions: int = 0,
        skipped_action_results: Sequence[Any] = (),
    ) -> None:
        self._episode = episode
        self._raw_api = episode.api
        self._allowed = allowed
        self._max_calls = int(max_calls)
        self._planning_horizon_actions = planning_horizon_actions
        self._action_calls = 0
        self._skip_actions = max(0, int(skip_actions))
        self._skipped_action_results = tuple(skipped_action_results)
        self._skipped_actions = 0
        self.calls: list[CallRecord] = []
        self._occ_counts: dict[tuple[str, int], int] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self._allowed:
            raise AttributeError(f"API has no attribute {name!r}")
        raw_fn = getattr(self._raw_api, name)

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            is_action = name not in {"get_ee_pose", "get_ee_wrench", "get_object_pose", "get_gripper_state"}
            if is_action and self._skipped_actions < self._skip_actions:
                i = self._skipped_actions
                self._skipped_actions += 1
                return self._skipped_action_results[i] if i < len(self._skipped_action_results) else None
            if len(self.calls) >= self._max_calls:
                raise EpisodeTerminated(f"max_calls ({self._max_calls}) exceeded")
            lineno = _policy_caller_lineno()
            key = (name, lineno)
            occ = self._occ_counts.get(key, 0) + 1
            self._occ_counts[key] = occ
            idx = len(self.calls)

            try:
                res = raw_fn(*args, **kwargs)
            except EpisodeTerminated as exc:
                obs = self._episode.snapshot()
                self.calls.append(
                    CallRecord(
                        index=idx,
                        name=name,
                        lineno=lineno,
                        occurrence=occ,
                        args=tuple(args),
                        kwargs=dict(kwargs),
                        result={"terminated": exc.reason},
                        obs=obs,
                        sim_time=float(getattr(self._episode, "sim_time", 0.0)),
                    )
                )
                raise

            obs = self._episode.snapshot()
            self.calls.append(
                CallRecord(
                    index=idx,
                    name=name,
                    lineno=lineno,
                    occurrence=occ,
                    args=tuple(args),
                    kwargs=dict(kwargs),
                    result=res,
                    obs=obs,
                    sim_time=float(getattr(self._episode, "sim_time", 0.0)),
                )
            )
            if is_action:
                self._action_calls += 1
                if self._planning_horizon_actions is not None and self._action_calls >= self._planning_horizon_actions:
                    raise PlanningHorizonReached(f"forecast horizon {self._planning_horizon_actions} actions reached")
            return res

        return _wrapped


def run_policy(
    source: str,
    episode: Episode,
    theta: dict[str, Any],
    seed: int = 0,
    max_lines: int = 50000,
    max_calls: int = 200,
    extra_primitives: Sequence[str] = (),
    evaluate: bool = True,
    capture_output: bool = False,
    planning_horizon_actions: int | None = None,
    skip_actions: int = 0,
    skipped_action_results: Sequence[Any] = (),
) -> Trace:
    """Execute one policy phase; set ``evaluate=False`` when another phase follows."""
    check_policy_source(source, extra_primitives=extra_primitives)
    code = compile(source, "<policy>", "exec")
    # A task episode owns task-specific permissions so every execution path
    # receives the same declared primitives, including development evaluators.
    allowed = (
        frozenset(PRIMITIVES)
        | frozenset(extra_primitives)
        | frozenset(getattr(episode, "extra_primitives", ()))
    )
    traced_api = _TracedAPI(episode=episode, allowed=allowed, max_calls=max_calls,
                            planning_horizon_actions=planning_horizon_actions,
                            skip_actions=skip_actions, skipped_action_results=skipped_action_results)

    ns: dict[str, Any] = {
        "__builtins__": {
            **SAFE_BUILTINS,
            **({"print": print} if capture_output else {}),
        },
        "math": math,
    }

    executed_lines: list[int] = []
    line_count = 0

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal line_count
        if frame.f_code.co_filename == "<policy>":
            if event == "line":
                line_count += 1
                if line_count > max_lines:
                    raise EpisodeTerminated(f"max_lines ({max_lines}) exceeded")
                executed_lines.append(int(frame.f_lineno))
            return _tracer
        return None

    t0 = time.perf_counter()
    error: str | None = None
    terminated_reason: str | None = None
    planning_horizon_reached = False
    stdout_buffer = StringIO() if capture_output else None
    stderr_buffer = StringIO() if capture_output else None

    prev_trace = sys.gettrace()
    try:
        with (
            redirect_stdout(stdout_buffer) if stdout_buffer is not None else nullcontext(),
            redirect_stderr(stderr_buffer) if stderr_buffer is not None else nullcontext(),
        ):
            sys.settrace(_tracer)
            exec(code, ns, ns)
            executed_lines.clear()
            run_fn = ns["run"]
            run_fn(traced_api)
    except PlanningHorizonReached:
        planning_horizon_reached = True
    except EpisodeTerminated as exc:
        terminated_reason = exc.reason
        error = exc.reason
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(prev_trace)

    wall_time = time.perf_counter() - t0
    sim_time = float(getattr(episode, "sim_time", 0.0))

    if not evaluate:
        success = False
        score = -1.0
        failure = error
        safety_violation = None
    else:
        eval_res = episode.evaluate()
        sim_time = float(getattr(episode, "sim_time", sim_time))
        if error is not None:
            # Any execution limit or runtime exception remains a failed rollout,
            # even when the final physical snapshot independently looks successful.
            success = False
            score = min(eval_res.score, -0.5) if eval_res.score >= 0.0 else eval_res.score
            failure = error
            safety_violation = eval_res.safety_violation
        else:
            success = eval_res.success
            score = eval_res.score
            failure = eval_res.failure or terminated_reason
            safety_violation = eval_res.safety_violation

    failure_call: int | None = None
    if not success and traced_api.calls:
        failure_call = len(traced_api.calls) - 1

    return Trace(
        protocol_version=getattr(episode, "protocol_version", "unversioned"),
        theta=dict(theta),
        seed=int(seed),
        calls=traced_api.calls,
        lines=executed_lines,
        success=success,
        score=float(score),
        failure=failure,
        failure_call=failure_call,
        safety_violation=safety_violation,
        error=error,
        sim_time=sim_time,
        wall_time=wall_time,
        stdout=stdout_buffer.getvalue() if stdout_buffer is not None else "",
        stderr=stderr_buffer.getvalue() if stderr_buffer is not None else "",
    )
