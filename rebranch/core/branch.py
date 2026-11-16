"""Branch synthesis prompt construction and path-divergence verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rebranch.api.primitives import API_DOC
from rebranch.core.conflict import DistinguishingObservable
from rebranch.core.lifting import hoist_inline_literals, lift_constants
from rebranch.core.retune import Retuner
from rebranch.exec.evaluator import Evaluator
from rebranch.exec.sandbox import StaticCheckError, check_policy_source
from rebranch.exec.trace import Trace
from rebranch.eval.metrics import format_sdk_calls, summarize_trace_for_llm
from rebranch.llm.client import LLMClient, extract_python_code
from rebranch.llm.prompts import (
    PHYSCAP_EXTRA_DOC,
    SYSTEM_PROMPT,
    build_branch_prompt as _prompt_build_branch,
)


@dataclass
class BranchVerifyResult:
    """Verification outcome for a branched candidate policy."""

    passed: bool
    traces: list[Trace]
    paths_diverged: bool
    all_succeeded: bool
    feedback: str


def format_conflict_summary(
    core_traces: Sequence[Trace],
    incompatible_pairs: Sequence[tuple[int, int]],
    case_solutions: Sequence[Any] | None = None,
) -> str:
    """Summarize conflicting cases using strictly SDK-observable trace data (no theta)."""
    lines: list[str] = []
    for idx, tr in enumerate(core_traces):
        status = "SUCCESS" if tr.success else "FAILED"
        last_call = tr.calls[-1].name if tr.calls else "none"
        sol_note = ""
        if case_solutions is not None and idx < len(case_solutions):
            sol = case_solutions[idx]
            if getattr(sol, "solvable", False) and getattr(sol, "best_values", None):
                kv = ", ".join(f"{k}={v}" for k, v in sorted(sol.best_values.items()))
                sol_note = f"; individually solvable with {kv}"
        lines.append(
            f"- Case {idx}: {status} (score={tr.score:.2f}, last_call={last_call}; "
            f"task_failure={tr.failure or 'none'}; execution_error={tr.error or 'none'}{sol_note})"
        )
    for idx, tr in enumerate(core_traces[:3]):
        if tr.calls:
            lines.append(f"SDK evidence, Case {idx}:\n{format_sdk_calls(tr, max_calls=8)}")
    if incompatible_pairs:
        pair_str = ", ".join(f"(Case {i}, Case {j})" for i, j in incompatible_pairs)
        lines.append(f"Candidate case pairs selected for branch synthesis: {pair_str}")
    return "\n".join(lines)


def format_trace_feedback(
    traces: Sequence[Trace], max_calls_per_trace: int = 8, max_examples: int = 3
) -> str:
    """Format compact SDK call traces and observations (strictly zero theta leakage)."""
    lines: list[str] = []
    lines.append(f"Trace counts: total={len(traces)}, success={sum(t.success for t in traces)}, failed={sum(not t.success for t in traces)}")
    counts: dict[tuple[str, str, str], int] = {}
    for tr in traces:
        key = (tr.failure or "none", tr.error or "none", tr.safety_violation or "none")
        counts[key] = counts.get(key, 0) + 1
    for (failure, error, safety), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"task_failure={failure}; execution_error={error}; safety_violation={safety}: {count}")
    failed_examples = [(i, tr) for i, tr in enumerate(traces) if not tr.success]
    examples = failed_examples[:max_examples] if failed_examples else list(enumerate(traces[:max_examples]))
    for idx, tr in examples:
        lines.append(f"[Failure Case {idx}]\n{summarize_trace_for_llm(tr, max_calls=max_calls_per_trace)}")
    return "\n".join(lines)


def format_observables_summary(
    observables: Sequence[DistinguishingObservable],
    obs_api_map: Mapping[str, str] | None = None,
) -> str:
    """Format ranked distinguishing SDK observables without any ground-truth theta."""
    if not observables:
        return "No strong single-call divergence observed; probe with zero_ee_wrench(), close_gripper(), or move_compliant()."
    lines: list[str] = []
    for obs in observables[:5]:
        api_hint = obs.api_expr
        if obs_api_map and obs.key in obs_api_map:
            api_hint = obs_api_map[obs.key]
        opp_tag = " [opposite signs]" if obs.opposite_sign else ""
        lines.append(
            f"- `{obs.key}` via `{api_hint}` after `{obs.call_name}` (line {obs.lineno}): "
            f"Case A={obs.val_a:.4f} vs Case B={obs.val_b:.4f} "
            f"(sep={obs.score:.2f} sigma{opp_tag})"
        )
    if observables[0].score >= 3.0:
        lines.append(
            "Hint: Threshold the top distinguishing observable directly after the probing "
            "call (midpoint between Case A and Case B) rather than noisy multi-step diffs."
        )
    return "\n".join(lines)


def build_branch_prompt(
    task_description: str,
    current_code: str,
    core_traces: Sequence[Trace],
    incompatible_pairs: Sequence[tuple[int, int]],
    observables: Sequence[DistinguishingObservable],
    *,
    case_solutions: Sequence[Any] | None = None,
    obs_api_map: Mapping[str, str] | None = None,
    failure_feedback: str = "",
    api_doc: str = API_DOC,
    physcap_primitives: bool = False,
) -> str:
    """Build a branch-synthesis prompt via `prompts.py` with zero theta leakage."""
    doc = f"{api_doc}\n\n{PHYSCAP_EXTRA_DOC}" if physcap_primitives else api_doc
    conflict_summary = format_conflict_summary(
        core_traces, incompatible_pairs, case_solutions=case_solutions
    )
    obs_summary = format_observables_summary(observables, obs_api_map=obs_api_map)
    return _prompt_build_branch(
        task_description=task_description,
        current_code=current_code,
        conflict_summary=conflict_summary,
        distinguishing_observables=obs_summary,
        failure_feedback=failure_feedback,
        api_doc=doc,
    )


def verify_branch(
    evaluator: Evaluator,
    candidate_source: str,
    core_cases: Sequence[Any],
    incompatible_pairs: Sequence[tuple[int, int]],
) -> BranchVerifyResult:
    """Verify `candidate_source` passes all `core_cases` and diverges on `incompatible_pairs`."""
    try:
        check_policy_source(
            candidate_source, extra_primitives=evaluator.extra_primitives
        )
    except StaticCheckError as exc:
        return BranchVerifyResult(
            passed=False,
            traces=[],
            paths_diverged=False,
            all_succeeded=False,
            feedback=f"Static sandbox check failed: {exc}",
        )

    traces = evaluator.run_many(candidate_source, core_cases)
    all_succeeded = all(tr.success for tr in traces)
    same_path_pairs = [
        (i, j)
        for i, j in incompatible_pairs
        if i < len(traces) and j < len(traces) and traces[i].path == traces[j].path
    ]
    paths_diverged = len(same_path_pairs) == 0

    fb_lines: list[str] = []
    if not all_succeeded or not paths_diverged:
        fb_lines.append(format_trace_feedback(traces))
    if not paths_diverged:
        pair_txt = ", ".join(f"(Case {i}, Case {j})" for i, j in same_path_pairs)
        fb_lines.append(
            f"Incompatible pair(s) {pair_txt} executed the exact same line path "
            "(trace_i.path == trace_j.path). Add an explicit sensory probe and `if / else` branch."
        )

    passed = all_succeeded and paths_diverged
    return BranchVerifyResult(
        passed=passed,
        traces=traces,
        paths_diverged=paths_diverged,
        all_succeeded=all_succeeded,
        feedback="\n".join(fb_lines),
    )


def synthesize_branch(
    evaluator: Evaluator,
    llm: LLMClient,
    task_description: str,
    current_code: str,
    core_cases: Sequence[Any],
    core_traces: Sequence[Trace],
    incompatible_pairs: Sequence[tuple[int, int]],
    observables: Sequence[DistinguishingObservable],
    *,
    case_solutions: Sequence[Any] | None = None,
    obs_api_map: Mapping[str, str] | None = None,
    retuner: Retuner | None = None,
    max_retries: int = 2,
    no_branch_verify: bool = False,
    no_lifting: bool = False,
    physcap_primitives: bool = False,
    initial_feedback: str = "",
) -> tuple[str, BranchVerifyResult]:
    """Call LLM to synthesize a branched policy and verify/retry up to `max_retries` times."""
    feedback = initial_feedback
    best_source = current_code
    best_ver = BranchVerifyResult(
        passed=False,
        traces=[],
        paths_diverged=False,
        all_succeeded=False,
        feedback="No candidate produced.",
    )
    best_rank = (-1, -1, -1e9)

    n_attempts = 1 if no_branch_verify else (1 + max(0, int(max_retries)))
    for _ in range(n_attempts):
        prompt = build_branch_prompt(
            task_description=task_description,
            current_code=current_code,
            core_traces=core_traces,
            incompatible_pairs=incompatible_pairs,
            observables=observables,
            case_solutions=case_solutions,
            obs_api_map=obs_api_map,
            failure_feedback=feedback,
            physcap_primitives=physcap_primitives,
        )
        from rebranch.llm.client import LLMBudgetExceededError
        try:
            resp = llm.complete(prompt, system=SYSTEM_PROMPT)
        except LLMBudgetExceededError:
            break  # Return the best measured branch for outer admission.
        cand = extract_python_code(resp.text)

        if no_branch_verify:
            try:
                check_policy_source(cand, extra_primitives=evaluator.extra_primitives)
                ver = BranchVerifyResult(
                    passed=True,
                    traces=[],
                    paths_diverged=True,
                    all_succeeded=True,
                    feedback="",
                )
                return cand, ver
            except StaticCheckError as exc:
                feedback = f"Static sandbox check failed: {exc}"
                continue

        ver = verify_branch(evaluator, cand, core_cases, incompatible_pairs)
        if ver.passed:
            return cand, ver

        # If the LLM wrote a branching structure (paths diverged) and at least one
        # case succeeded, do a lightweight constant polish before spending a retry.
        # Skip retuning when 0 cases succeed (logic/coordinate error needs LLM feedback).
        if (
            ver.paths_diverged
            and any(t.success for t in ver.traces)
            and retuner is not None
            and not no_lifting
        ):
            cand_lifted = cand
            consts = lift_constants(cand_lifted)
            if not consts:
                cand_lifted = hoist_inline_literals(cand_lifted)
                consts = lift_constants(cand_lifted)
            if consts:
                polish_retuner = Retuner(
                    evaluator,
                    max_iters=min(retuner.max_iters, 5),
                    popsize=min(retuner.popsize or 8, 8),
                    margin_target=0.01,
                    sigma0=retuner.sigma0,
                    seed=retuner.seed,
                    max_rollouts=120,
                    budget_reserve=retuner.budget_reserve,
                )
                opt = polish_retuner.optimize(cand_lifted, consts, core_cases)
                ver_opt = verify_branch(
                    evaluator, opt.best_source, core_cases, incompatible_pairs
                )
                if ver_opt.passed:
                    return opt.best_source, ver_opt
                if sum(t.success for t in ver_opt.traces) > sum(
                    t.success for t in ver.traces
                ):
                    cand = opt.best_source
                    ver = ver_opt

        n_ok = sum(1 for t in ver.traces if t.success)
        mean_sc = (
            sum(t.score for t in ver.traces) / len(ver.traces) if ver.traces else -1.0
        )
        rank = (int(ver.paths_diverged), n_ok, mean_sc)
        if rank > best_rank:
            best_rank = rank
            best_source = cand
            best_ver = ver

        feedback = (
            f"Previous Candidate:\n```python\n{cand.strip()}\n```\n\n"
            f"Verification Output:\n{ver.feedback}"
        )

    return best_source, best_ver
