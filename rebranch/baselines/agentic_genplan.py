"""Tool-driven AgenticGenPlan platform adaptation for the local RobotAPI tasks."""
from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import io
import json
import math
import sys
import statistics
import time
from typing import Any

import numpy as np

from rebranch.api.primitives import API_DOC
from rebranch.baselines import BaselineResult
from rebranch.api.primitives import EpisodeTerminated
from rebranch.exec.sandbox import SAFE_BUILTINS, StaticCheckError, _TracedAPI, check_policy_source
from rebranch.exec.trace import Trace
from rebranch.llm.client import LLMBudgetExceededError, LLMClient, LLMResponse
from rebranch.sim.backend import TaskBackend

_NUMERIC_STDLIB_MODULES = {"math", "statistics", "itertools", "collections",
                           "functools", "operator", "decimal", "fractions"}
_AGENTIC_MODULES = _NUMERIC_STDLIB_MODULES | {"numpy", "scipy"}
_UNAVAILABLE_MODULES = {"os", "sys", "subprocess", "socket", "pathlib", "shutil",
                        "ctypes", "multiprocessing", "pickle", "tempfile", "importlib",
                        "builtins", "io", "inspect", "gc"}


AGENTIC_SYSTEM = """You are an autonomous experimenter. Use the available tools to inspect the task, write and run diagnostic RobotAPI policies, inspect complete SDK feedback, write analysis scripts, revise policies, and explicitly freeze one final policy. Choose development cases and experiments yourself. Do not infer or request hidden parameters. A run's outcome and SDK trace are observations; its parameter vector is withheld. Every experiment costs one rollout and every tool decision costs one LLM request. Return exactly one JSON object with a tool name and arguments."""


def _public_json_value(value: Any) -> Any:
    """Normalize NumPy values/containers to the public JSON data model."""
    if isinstance(value, np.ndarray):
        return _public_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _public_json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _public_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_json_value(item) for item in value]
    return value


def _safe_analysis(script: str, data: Any) -> Any:
    """Run an agent-authored diagnostic function with numerical modules, no I/O imports."""
    import scipy
    def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        root = name.split('.')[0]
        if level or root not in _AGENTIC_MODULES or root in _UNAVAILABLE_MODULES:
            raise ImportError(f"diagnostic import not available: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    scope: dict[str, Any] = {
        "__builtins__": {k: getattr(builtins, k) for k in (
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "len", "list", "max", "min", "range", "round", "set", "sorted",
            "str", "sum", "tuple", "zip", "print", "isinstance", "issubclass",
            "Exception", "ValueError", "TypeError", "NameError", "AssertionError",
            "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError", "StopIteration"
        )} | {"__import__": safe_import},
        "data": data,
    }
    exec(compile(script, "<agent-analysis>", "exec"), scope, scope)
    fn = scope.get("analyze")
    if not callable(fn):
        raise ValueError("analysis script must define analyze(data)")
    result = _public_json_value(fn(data))
    json.dumps(result, allow_nan=False)
    return result


def _run_agentic_policy(source: str, episode, theta: dict[str, Any], *, seed: int, max_calls: int) -> Trace:
    """Agentic-only executor for stdlib/numpy/scipy policies with local RobotAPI caps."""
    try:
        tree = ast.parse(source, filename="<agentic-policy>")
    except SyntaxError as exc:
        raise StaticCheckError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    disallowed = (imports - _AGENTIC_MODULES) | (imports & _UNAVAILABLE_MODULES)
    if disallowed:
        raise StaticCheckError(f"Agentic policy imports are not available: {sorted(disallowed)}")

    import copy
    checked_tree = copy.deepcopy(tree)
    class _StripImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node)
    checked_tree = _StripImports().visit(checked_tree)
    ast.fix_missing_locations(checked_tree)
    check_policy_source(ast.unparse(checked_tree))
    # The executable tree is untouched; function-local imports and globals retain
    # normal Python semantics in this method-specific namespace.
    code = compile(tree, "<policy>", "exec")

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split('.')[0]
        if level or root not in _AGENTIC_MODULES or root in _UNAVAILABLE_MODULES:
            raise ImportError(f"Agentic policy import is unavailable: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    builtins_ns = dict(SAFE_BUILTINS)
    builtins_ns.update({"__import__": safe_import, "print": print,
                        "isinstance": isinstance, "issubclass": issubclass,
                        "Exception": Exception, "ValueError": ValueError,
                        "TypeError": TypeError, "NameError": NameError,
                        "ArithmeticError": ArithmeticError, "AssertionError": AssertionError})
    from rebranch.api.primitives import PRIMITIVES
    api = _TracedAPI(
        episode,
        frozenset(PRIMITIVES) | frozenset(getattr(episode, "extra_primitives", ())),
        max_calls,
    )
    stdout, stderr = io.StringIO(), io.StringIO()
    error = None
    lines: list[int] = []
    line_count = 0
    def tracer(frame, event, arg):
        nonlocal line_count
        if frame.f_code.co_filename == "<policy>" and event == "line":
            line_count += 1
            if line_count > 50000:
                raise EpisodeTerminated("max_lines (50000) exceeded")
            lines.append(frame.f_lineno)
        return tracer if frame.f_code.co_filename == "<policy>" else None
    before = time.perf_counter()
    old_trace = sys.gettrace()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            sys.settrace(tracer)
            ns = {"__builtins__": builtins_ns, "math": math, "statistics": statistics}
            exec(code, ns, ns)
            lines.clear()
            ns["run"](api)
    except Exception as exc:
        error = f"{type(exc).__name__}: {getattr(exc, 'reason', str(exc))}"
    finally:
        sys.settrace(old_trace)
    result = episode.evaluate()
    if error:
        success, score, failure = False, min(float(result.score), -0.5), error
    else:
        success, score, failure = result.success, result.score, result.failure
    return Trace(theta=dict(theta), seed=seed, calls=api.calls, lines=lines,
        success=success, score=score, failure=failure,
        failure_call=(len(api.calls)-1 if not success and api.calls else None),
        safety_violation=result.safety_violation, error=error,
        sim_time=float(getattr(episode, "sim_time", 0.0)), wall_time=time.perf_counter()-before,
        protocol_version=getattr(episode, "protocol_version", "unversioned"),
        stdout=stdout.getvalue(), stderr=stderr.getvalue())


def _run_agentic_policy_source_check(source: str) -> None:
    try:
        tree = ast.parse(source, filename="<agentic-policy>")
    except SyntaxError as exc:
        raise StaticCheckError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split('.')[0] not in _AGENTIC_MODULES or a.name.split('.')[0] in _UNAVAILABLE_MODULES for a in node.names):
            raise StaticCheckError("Agentic policy imports must use the available standard library, NumPy, or SciPy modules")
        if isinstance(node, ast.ImportFrom) and (not node.module or node.module.split('.')[0] not in _AGENTIC_MODULES or node.module.split('.')[0] in _UNAVAILABLE_MODULES):
            raise StaticCheckError("Agentic policy imports must use the available standard library, NumPy, or SciPy modules")
    checked_tree = copy.deepcopy(tree)
    class _StripImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node)
    checked_tree = _StripImports().visit(checked_tree)
    ast.fix_missing_locations(checked_tree)
    check_policy_source(ast.unparse(checked_tree))

def _trace_public(trace: Trace) -> dict[str, Any]:
    return {
        "success": trace.success,
        "score": trace.score,
        "failure": trace.failure,
        "execution_error": trace.error,
        "sdk_calls": [
            {"name": c.name, "args": c.args, "kwargs": c.kwargs,
             "result": c.result, "obs": c.obs, "sim_time": c.sim_time}
            for c in trace.calls
        ],
        "sim_time": trace.sim_time,
        "stdout": trace.stdout,
        "stderr": trace.stderr,
    }


class AgenticGenPlanAgent:
    """Frozen RobotAPI policy; all language-model/tool use is development-only."""
    def __init__(self, source: str, *, max_calls: int = 200):
        self.source = source
        self.max_calls = max_calls

    def evaluate_episode(self, backend: TaskBackend, theta: dict[str, Any], seed: int = 0):
        ep = backend.make_episode(theta, seed=seed)
        trace = _run_agentic_policy(self.source, ep, theta, seed=seed, max_calls=self.max_calls)
        return trace, {"policy_source": self.source, "runtime_llm_calls": 0,
                       "runtime_sim_rollouts": 0, "runtime_sim_time_s": 0.0,
                       "sim_predicted_success": None}


def run_agentic_genplan(
    backend: TaskBackend,
    llm: LLMClient,
    *, seed: int = 0, max_rollouts: int = 2000, max_llm_calls: int = 8,
    n_dev_cases: int = 24, max_policy_calls: int = 200,
) -> BaselineResult:
    """Autonomous inspect/experiment/analyze/revise/freeze loop on sampled dev cases."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dev_cases = [backend.param_space.sample(rng) for _ in range(max(1, n_dev_cases))]
    calls = 0
    prompt_tokens = completion_tokens = rollouts = 0
    policy_source: str | None = None
    test_source: str | None = None
    frozen = False
    stop_reason = "llm_budget_exhausted"
    events: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    last_error: str | None = None

    def request(prompt: str) -> LLMResponse:
        nonlocal calls, prompt_tokens, completion_tokens, stop_reason
        if calls >= max_llm_calls:
            raise StopIteration("LLM request budget exhausted")
        calls += 1
        try:
            response = llm.complete(prompt, system=AGENTIC_SYSTEM, temperature=0.2)
        except LLMBudgetExceededError as exc:
            calls -= 1
            events.append({"budget_rejection": True, "error": str(exc), "request_counted": False})
            stop_reason = "external_llm_budget_exhausted"
            raise StopIteration("external LLM budget exhausted") from exc
        except StopIteration as exc:
            calls -= 1
            events.append({"budget_rejection": True, "error": str(exc), "request_counted": False})
            stop_reason = "external_llm_budget_exhausted"
            raise StopIteration("external LLM budget exhausted") from exc
        except Exception as exc:
            events.append({"request": calls, "error": f"{type(exc).__name__}: {exc}"})
            raise
        prompt_tokens += int(response.prompt_tokens)
        completion_tokens += int(response.completion_tokens)
        events.append({"request": calls, "prompt": prompt, "response": response.text,
                       "prompt_tokens": response.prompt_tokens,
                       "completion_tokens": response.completion_tokens})
        return response

    def tool_result(action: dict[str, Any]) -> dict[str, Any]:
        nonlocal policy_source, test_source, rollouts, frozen, stop_reason, last_error
        tool = action.get("tool")
        if not isinstance(tool, str):
            return {"error": "tool must be a string"}
        if tool == "inspect_task":
            return {"task_description": backend.task_description, "robot_api": API_DOC,
                    "available_tools": ["write_policy", "write_test", "run_experiment", "run_test",
                        "inspect_run", "run_analysis", "freeze_policy"]}
        if tool in {"write_policy", "write_test"}:
            source = action.get("source")
            if not isinstance(source, str):
                return {"accepted": False, "error": "source must be a Python string"}
            try:
                _run_agentic_policy_source_check(source)
            except (StaticCheckError, SyntaxError) as exc:
                last_error = str(exc)
                return {"accepted": False, "error": last_error}
            if tool == "write_policy":
                policy_source = source
                snapshots.append({"version": len(snapshots), "kind": "policy", "source": source})
            else:
                test_source = source
                snapshots.append({"version": len(snapshots), "kind": "test", "source": source})
            return {"accepted": True, "kind": tool, "sha256": __import__("hashlib").sha256(source.encode()).hexdigest()}
        if tool in {"run_experiment", "run_test"}:
            source_kind = action.get("source_kind", "test" if tool == "run_test" else "policy")
            if not isinstance(source_kind, str) or source_kind not in {"policy", "test"}:
                return {"error": "source_kind must be 'policy' or 'test'"}
            selected_source = test_source if source_kind == "test" else policy_source
            if selected_source is None:
                return {"error": "write the selected source before running it"}
            if rollouts >= max_rollouts:
                stop_reason = "rollout_budget_exhausted"
                return {"error": stop_reason}
            indices = action.get("case_indices")
            if not isinstance(indices, list) or any(type(idx) is not int for idx in indices):
                return {"error": "case_indices must be a list of integer development indices"}
            invalid = [idx for idx in indices if idx < 0 or idx >= len(dev_cases)]
            if invalid:
                return {"error": f"development case indices out of range: {invalid}"}
            result_ids = []
            summaries = []
            for idx in indices:
                if rollouts >= max_rollouts:
                    break
                ep = backend.make_episode(dev_cases[idx], seed=seed + 1000 + idx)
                tr = _run_agentic_policy(selected_source, ep, dev_cases[idx], seed=seed + 1000 + idx,
                                         max_calls=max_policy_calls)
                rollouts += 1
                record = {"run_id": len(runs), "case_index": idx, **_trace_public(tr)}
                runs.append(record)
                result_ids.append(record["run_id"])
                summaries.append({"run_id": record["run_id"], "case_index": idx,
                                  "success": tr.success, "score": tr.score,
                                  "failure": tr.failure, "execution_error": tr.error})
            return {"run_ids": result_ids, "results": summaries, "source_kind": source_kind,
                    "remaining_rollouts": max_rollouts - rollouts}
        if tool == "inspect_run":
            idx = action.get("run_id")
            if type(idx) is not int:
                return {"error": "run_id must be an integer returned by an experiment"}
            return runs[idx] if 0 <= idx < len(runs) else {"error": "unknown run_id"}
        if tool == "run_analysis":
            try:
                script = action.get("script")
                if not isinstance(script, str):
                    return {"error": "script must be Python source as a string"}
                output = _safe_analysis(script, copy.deepcopy(runs))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                return {"error": last_error}
            analyses.append({"script": action.get("script", ""), "result": output})
            return {"result": output}
        if tool == "freeze_policy":
            if policy_source is None:
                return {"error": "cannot freeze before writing a valid policy"}
            frozen = True
            stop_reason = "explicit_freeze"
            return {"frozen": True, "sha256": __import__("hashlib").sha256(policy_source.encode()).hexdigest()}
        return {"error": f"unknown tool: {tool!r}"}

    transcript: list[dict[str, Any]] = []
    schemas = {
        "inspect_task": {"args": {}, "returns": "task description, RobotAPI specification, no draft implementation, and tool capabilities"},
        "write_policy": {"args": {"source": "Python source defining run(api); math/numpy/scipy imports allowed"}, "returns": "accepted/error and source sha256"},
        "write_test": {"args": {"source": "Python diagnostic/test policy defining run(api)"}, "returns": "accepted/error and source sha256"},
        "run_experiment": {"args": {"source_kind": "policy|test (default policy)", "case_indices": "required list of integers in [0, n_dev_cases)"}, "returns": "run_ids, per-case success/score/failure/execution_error summaries, and remaining_rollouts"},
        "run_test": {"args": {"case_indices": "required list of integers in [0, n_dev_cases); runs write_test source"}, "returns": "run_ids, per-case success/score/failure/execution_error summaries, and remaining_rollouts"},
        "inspect_run": {"args": {"run_id": "integer returned by run_experiment/run_test"}, "returns": "full SDK trace, task result, stdout/stderr, no hidden theta"},
        "run_analysis": {"args": {"script": "Python source defining analyze(data); can import math/statistics/numpy/scipy"}, "returns": "analysis result or error"},
        "freeze_policy": {"args": {}, "returns": "frozen=true and source sha256"},
    }
    prompt = json.dumps({"task": backend.task_description, "robot_api": API_DOC,
        "instruction": "Autonomously inspect, write and run diagnostics, select development cases, revise, then explicitly freeze a generalized policy.",
        "tools": schemas, "n_dev_cases": len(dev_cases), "legal_case_indices": [0, len(dev_cases)-1],
        "remaining_rollouts": max_rollouts-rollouts, "remaining_llm_requests": max_llm_calls-calls,
        "interaction_history": transcript}, ensure_ascii=False, indent=2)
    while calls < max_llm_calls and not frozen:
        try:
            response = request(prompt)
        except StopIteration:
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            stop_reason = "llm_error"
            break
        try:
            action = json.loads(response.text)
            if not isinstance(action, dict) or "tool" not in action:
                raise ValueError("response must be a JSON object with tool")
        except Exception as exc:
            last_error = f"invalid tool response: {exc}"
            tool_out = {"error": last_error}
            action = {"tool": "invalid"}
        else:
            tool_out = tool_result(action)
        transcript.append({"action": action, "tool_result": tool_out})
        prompt = json.dumps({"task": backend.task_description, "robot_api": API_DOC, "tools": schemas,
            "n_dev_cases": len(dev_cases), "legal_case_indices": [0, len(dev_cases)-1],
            "remaining_rollouts": max_rollouts-rollouts,
            "remaining_llm_requests": max_llm_calls-calls,
            "interaction_history": transcript,
            "current_policy_source": policy_source, "current_test_source": test_source,
            "next": "Continue diagnosis/revision using the full history, or explicitly freeze_policy."},
            ensure_ascii=False, indent=2)
    if not frozen and rollouts >= max_rollouts and stop_reason != "external_llm_budget_exhausted":
        stop_reason = "rollout_budget_exhausted"
    if not frozen and calls >= max_llm_calls and stop_reason != "external_llm_budget_exhausted":
        stop_reason = "llm_budget_exhausted"
    if policy_source is None:
        raise RuntimeError(f"AgenticGenPlan produced no valid committed policy before stop ({stop_reason}); no draft fallback is assigned")
    if not frozen:
        frozen = True
        stop_reason = "budget_stop" if stop_reason in {"llm_budget_exhausted", "rollout_budget_exhausted"} else stop_reason
    return BaselineResult(
        method="agentic_genplan", policy_source=policy_source, n_rollouts=rollouts,
        dev_llm_calls=calls, dev_prompt_tokens=prompt_tokens,
        dev_completion_tokens=completion_tokens,
        agent=AgenticGenPlanAgent(policy_source, max_calls=max_policy_calls),
        round_logs=events + [{"tool_transcript": transcript}],
        program_snapshots=snapshots, wall_time_s=time.perf_counter()-t0,
        extra={"identity": "platform_adaptation", "frozen": frozen,
               "freeze_kind": "explicit_agent_freeze" if stop_reason == "explicit_freeze" else "budget_stop",
               "stop_reason": stop_reason, "last_error": last_error,
               "n_development_cases": len(dev_cases), "n_experiments": rollouts,
               "development_runs": runs, "analysis_scripts": analyses,
               "written_test_source": test_source,
               "runtime_llm_calls": 0, "runtime_sim_rollouts": 0,
               "tool_events": transcript, "policy_freedom": "LLM authors multi-step policy and diagnostic programs, chooses cases/iterations, and freezes explicitly; execution and analysis imports are limited to math/statistics/itertools/collections/functools/operator/decimal/fractions plus numpy/scipy and preserve API caps. Filesystem/network/source-inspection modules are unavailable; this is an import allowlist, not a security sandbox."},
    )
